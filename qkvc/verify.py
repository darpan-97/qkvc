"""The proof, for any built task: the model writes exactly what the task's rules say, for every
valid input of any length (S, B to G). The rules are checked against the task's reference (A).

    python -m qkvc verify TASK

  A  the task's rules equal its reference: every full group (when there are at most 2^24) and
     sampled whole inputs; groups are independent
  S  the stream: every symbol an output reads is the same distance back in every group, so one
     distance per (place, fetch) is right for an input of any length
  B  where each head looks, read off Q and K for every phase
  C  what each head brings, read off V and O
  D  a bound on what attention leaks, over a window of W tokens
  E  the MLP and output layer give the rules' token on every case a valid input can produce
  F  the leak cannot change a token (interval bounds through the MLP and output layer)
  G  the whole answer, by induction over the stream
"""
import itertools
import os
import random
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qkvc.build import period, query_slot, stream_index      # noqa: E402


class NotProved(Exception):
  pass


def link(say, name, ok, detail):
  say(f'  {"holds" if ok else "FAILS"}  {name}')
  say(textwrap.fill(detail, 92, initial_indent=' ' * 9, subsequent_indent=' ' * 9))
  if not ok:
    raise NotProved(name)


def fetch(task, m, names, n, written=()):
  """(kind, value) of each symbol output n reads, from the input `names` or (for a fetch in
  from_outputs) the outputs `written` so far: 'symbol' with its value, a flag token's kind, or
  'missing' before the start or past the end."""
  symbols = {s.kind: s for s in task.symbols}
  flags = {t: k for k, t in task.flag_tokens.items()}
  out = []
  for i, p in enumerate(m.sources(n)):
    seq, alphabet = (written, symbols[m.outputs]) if m.reads_output(i) else (names, symbols[m.inputs])
    if p < 0 or p >= len(seq):
      out.append(('missing', 0))
    elif seq[p] in alphabet.names:
      out.append(('symbol', alphabet.names.index(seq[p])))
    else:
      out.append((flags[seq[p]], 0))
  return out


def rule_says(m, place, kinds):
  for r in m.rules:
    if r.place == place and all(kinds[m.fetch.index(head)] == flag for head, flag in r.conditions):
      return r.writes
  return None


def rules_answer(task, mode, names):
  """The output token names the task's rules give (END not included), and every (place, kinds)
  case met on the way."""
  m = task.mode(mode)
  out_names = {s.kind: s for s in task.symbols}[m.outputs].names
  out, cases = [], []
  groups = -(-max(len(names) - m.header, 0) // m.group[0]) + 2
  for n in range(groups * m.group[1] + m.extra + 1):
    place = n % m.group[1]
    got = fetch(task, m, names, n, out)
    kinds = tuple(k for k, _ in got)
    cases.append((place, kinds))
    says = rule_says(m, place, kinds)
    if says == 'end':
      return out, cases
    out.append(m.pad if says == 'pad' else out_names[m.combine(*(v for _, v in got), place)])
  raise NotProved(f'the rules never end {mode} on an input of {len(names)} symbols')


# ---- A: the rules equal the reference ----------------------------------------------------------
def prove_rules(task, say, limit=1 << 24):
  for m in task.modes:
    symbols = {s.kind: s for s in task.symbols}
    alphabet, out_names = np.array(symbols[m.inputs].names), np.array(symbols[m.outputs].names)
    size_in, size_out = m.group
    h = m.header
    width = h + size_in                                    # the header and the first group
    unit = 'groups' if not h else 'headers and groups'

    def reference_rows(digits):
      """The reference's outputs for each row of `digits` (a header and one group): (rows, size_out)."""
      names = alphabet[digits]
      if not h:                                            # groups simply follow each other
        return np.array(task.reference(m.name, m.write(names.reshape(-1).tolist()))).reshape(-1, size_out)
      return np.array([task.reference(m.name, m.write(row)) for row in names.tolist()])

    def rules_rows(digits, j):
      """The rules' output j for each row: a position past the first group reads as 0."""
      values = [digits[:, i] for i in range(width)] + [digits[:, 0] * 0]
      return out_names[m.combine(*(values[min(p, width)] for p in m.sources(j)), j)]

    if m.from_outputs:
      checked, length = 0, 0
      while len(alphabet) ** (length + 1) <= 1 << 16:
        length += 1
        for digits in itertools.product(range(len(alphabet)), repeat=length):
          names = [str(alphabet[d]) for d in digits]
          try:
            data = m.write(names)
          except ValueError:                               # not a valid input of this length
            continue
          if m.read(data) != names:
            continue
          if rules_answer(task, m.name, names)[0] != task.reference(m.name, data):
            link(say, f'A  {m.name}: rules = reference on every short input', False, f'input {data!r}')
          checked += 1
      link(say, f'A  {m.name}: rules = reference on every short input', True,
           f'every valid input of up to {length} {m.inputs}s: {checked:,} inputs. A carry runs along the whole '
           f'chain, so groups are not independent: longer inputs are sampled below, and the proof of the '
           f'model (B-G) covers every length')
      run_samples(task, m, size_in, say)
      continue
    total = len(alphabet) ** width
    if total <= limit:
      wrong = 0
      for g0 in range(0, total, 1 << 18):
        g = np.arange(g0, min(total, g0 + (1 << 18)), dtype=np.int64)
        digits = np.stack([(g // len(alphabet) ** (width - 1 - i)) % len(alphabet) for i in range(width)], 1)
        got = reference_rows(digits)
        wrong += int(sum((got[:, j] != rules_rows(digits, j)).sum() for j in range(size_out)))
      link(say, f'A  {m.name}: rules = reference on every full group', wrong == 0,
           f'all {total:,} {unit} of {width} {m.inputs}{"s" if width > 1 else ""}, {wrong} wrong')
    else:                                                  # every value of what each output reads
      fill, wrong, checked = np.random.default_rng(3), 0, 0
      for j in range(size_out):
        inside = sorted({p for p in m.sources(j) if p < width})
        combos = np.array(list(itertools.product(range(len(alphabet)), repeat=len(inside))))
        for _ in range(2):
          digits = fill.integers(0, len(alphabet), (len(combos), width))
          digits[:, inside] = combos
          wrong += int((reference_rows(digits)[:, j] != rules_rows(digits, j)).sum())
          checked += len(combos)
      link(say, f'A  {m.name}: rules = reference on every value each output reads', wrong == 0,
           f'{total:,} full {unit} is too many to list, so for every place: every value of the inputs it '
           f'reads, the rest random (twice); {checked:,} checked, {wrong} wrong')
    rng_v = np.random.default_rng(4)                       # a source past the first group must not matter
    for j in range(size_out):
      for i, p in enumerate(m.sources(j)):
        if p >= width:
          vals = [rng_v.integers(0, len(alphabet), 4096) for _ in m.fetch]
          base = [v.copy() for v in vals]
          base[i] = base[i] * 0
          if not np.array_equal(m.combine(*vals, j), m.combine(*base, j)):
            link(say, f'A  {m.name}: groups are independent', False, f'output {j} depends on the next group')
    shift = lambda n: tuple(p if p < h else p + size_in for p in m.sources(n))
    ok = all(tuple(m.sources(n + size_out)) == shift(n) for n in range(1 << 12))
    link(say, f'A  {m.name}: groups are independent', ok,
         f'output n + {size_out} reads one group further on than output n' + (', and the same header' if h else '')
         + ', for every n')
    run_samples(task, m, size_in, say)


def run_samples(task, m, size_in, say):
  """Rules = reference on sampled valid inputs of every length up to 8 groups, endings included."""
  rng, inputs = random.Random(1), 0
  for length in range(8 * size_in):
    for _ in range(25):
      data = m.sample(rng, length)
      names = m.read(data)
      got, _ = rules_answer(task, m.name, names)
      if got != task.reference(m.name, data):
        link(say, f'A  {m.name}: rules = reference on whole inputs', False, f'input {data!r}')
      inputs += 1
  link(say, f'A  {m.name}: rules = reference on whole inputs', True,
       f'{inputs:,} sampled valid inputs, every length up to {8 * size_in - 1} units, endings included')

def reachable(task, m):
  """Every (place, kinds of the fetched inputs) a valid input meets, from sampled inputs of every
  length up to 12 groups; checked to have stopped growing after 8."""
  rng = random.Random(2)
  def walk(groups):
    seen = set()
    for length in range(groups * m.group[0]):
      for _ in range(25):
        names = m.read(m.sample(rng, length))
        seen |= set(rules_answer(task, m.name, names)[1])
    return seen
  small, large = walk(8), walk(12)
  if small != large:
    raise NotProved(f'{m.name}: the cases were still growing')
  return sorted(large)


def movement(built):
  """If every number attention wrote is off by at most delta, every score moves at most L x delta."""
  S, mlp = built.S, built.mlp
  noisy = np.zeros(S.size)
  for name in S.blocks:
    if name.startswith('mode_') or '.' in name:
      noisy[S[name]] = 1
  unit_err = np.abs(mlp.W_in).T @ noisy
  out_err = np.abs(mlp.W_out).T @ unit_err + noisy
  return (np.abs(built.U).T @ out_err).max()


class Line:
  """g * a + b for a group number g >= 0: what sources() makes of an output index n = size * g + j.
  Every operation is exact for every g or it stops, so a source that is not one fixed distance per
  group cannot pass."""

  def __init__(self, a, b):
    self.a, self.b = a, b

  def _int(self, k):
    if isinstance(k, Line):
      if k.a:
        raise NotProved('S  sources() multiplies or divides by the output index')
      return k.b
    return k

  def __add__(self, k):
    return Line(self.a + k.a, self.b + k.b) if isinstance(k, Line) else Line(self.a, self.b + k)

  def __sub__(self, k):
    return self + (-k)

  def __neg__(self):
    return Line(-self.a, -self.b)

  def __mul__(self, k):
    k = self._int(k)
    return Line(self.a * k, self.b * k)

  def __floordiv__(self, k):
    k = self._int(k)
    if self.a % k:
      raise NotProved(f'S  sources() divides by {k}, which does not divide the group step')
    return Line(self.a // k, self.b // k)

  def __mod__(self, k):
    k = self._int(k)
    if self.a % k:
      raise NotProved(f'S  sources() takes the index mod {k}, which does not divide the group step')
    return Line(0, self.b % k)

  def __index__(self):
    return self._int(self)

  __radd__ = __add__
  __rmul__ = __mul__

  def __rsub__(self, k):
    return (-self) + k

  def __rfloordiv__(self, k):
    return Line(0, k // self._int(self))

  def __rmod__(self, k):
    return Line(0, k % self._int(self))


def prove_stream(r, say):
  """S: for each mode, place j and fetch i, the stream position output (g, j) reads is g * T + c
  with c fixed: sources() is run on the output index as a line in g, not on numbers."""
  for m in r.task.modes:
    T, (size_in, size_out), h = period(m), m.group, m.header
    for j in range(size_out):
      n = Line(size_out, j)
      for i, p in enumerate(m.sources(n)):
        p = p if isinstance(p, Line) else Line(0, p)
        if m.reads_output(i):
          if p.a != size_out:
            link(say, f'S  {m.name}: one distance per place', False, f'place {j} reads an output that moves {p.a} a group')
          at = Line(T, T * (p.b // size_out) + 1 + h + size_in + p.b % size_out)
        elif p.a == 0 and 0 <= p.b < h:                     # the header, copied into every group
          at = Line(T, 1 + p.b)
        elif p.a == size_in:
          q = p.b - h
          at = Line(T, T * (q // size_in) + 1 + h + q % size_in)
        else:
          link(say, f'S  {m.name}: one distance per place', False,
               f'place {j} fetch {m.fetch[i]} moves {p.a} inputs a group, not {size_in}')
        d = query_slot(m, j) - at.b                         # the query is g * T + query_slot
        if d < 0:                                           # in the next group, not yet in the stream
          ok = r.distances[m.name][(j, i)] is None and not m.reads_output(i) and not any(
              rule.place == j and m.fetch[i] in dict(rule.conditions) for rule in m.rules)
          link(say, f'S  {m.name} place {j}: {m.fetch[i]} is not needed', ok,
               f'it is in the next group, so the model reads it as missing; no rule at place {j} looks at it, '
               f'and A showed its value does not change the output')
          continue
        if d != r.distances[m.name][(j, i)] or d >= r.W:
          link(say, f'S  {m.name}: one distance per place', False,
               f'place {j} fetch {m.fetch[i]}: {d} back, the model uses {r.distances[m.name][(j, i)]}')
        for g in range(3):                                  # the line agrees with the stream's own layout
          if g * T + query_slot(m, j) - stream_index(m, g * size_out + j, i, g) != d:
            link(say, f'S  {m.name}: one distance per place', False, f'place {j} fetch {m.fetch[i]}, group {g}')
    link(say, f'S  {m.name}: one distance per place', True,
         f'sources() run on n = {size_out}g + j for a symbolic group g: every symbol output (g, j) reads is a '
         f'fixed distance back, under the window of {r.W}, for every g; a group {T} tokens long')
  link(say, 'S  every phase is a group place', r.P % np.lcm.reduce([period(m) for m in r.task.modes]) == 0 and r.W <= r.P,
       f'{r.P} phases, a multiple of every group length, so a token\'s phase gives its place in the group; '
       f'a window of {r.W} <= {r.P}, so within it no two tokens share a phase')


def prove_attention(r, say):
  task, S, P = r.task, r.S, r.P
  rows = lambda W: set(np.flatnonzero(np.abs(W).sum(1)))
  block = lambda name: set(range(S[name].start, S[name].stop))
  heads = {h.name: h for h in r.heads}
  mode_rows = {S.at(f'is_{m.flag}') for m in task.modes}
  phases = [S.at('phase', p) for p in range(P)]
  bits, flags = task.value_bits, list(task.flag_tokens)
  for m in task.modes:
    T = period(m)
    for i, label in enumerate(m.fetch):
      name = f'{m.flag}_{label}'
      h = heads[name]
      ok = rows(h.Q) <= block('phase') | {S.at('one')} and rows(h.K) <= block('phase') | mode_rows
      keys = h.K[phases]                                    # the key of a token at each phase
      for j in range(m.group[1]):
        d = r.distances[m.name][(j, i)]
        for phase in range(query_slot(m, j), P, T):
          q = h.Q[S.at('phase', phase)] + h.Q[S.at('one')]
          want = np.zeros(P) if d is None else (np.arange(P) == (phase - d) % P).astype(float)
          ok &= np.array_equal(q @ keys.T, want) and all(q @ h.K[f] == 0.5 for f in mode_rows)
      far = ', '.join('-' if r.distances[m.name][(j, i)] is None else str(r.distances[m.name][(j, i)])
                      for j in range(m.group[1]))
      link(say, f'B  {name} looks {far} back', ok,
           f'at every phase where it is used: score 1 at the token that far back, 0.5 at a MODE token, 0 at '
           f'every other token (- : only the MODE token): {P} x {P} phase pairs')
      ov, want = h.V @ h.O, np.zeros((S.size, S.size))
      for b in range(bits):
        want[S.at('bits', b), S.at(f'{name}.bits', b)] = 1
      for kind in flags:
        want[S.at(f'is_{kind}'), S.at(f'{name}.{kind}')] = 1
      for f in mode_rows | {S.at('is_none')}:
        want[f, S.at(f'{name}.missing')] = 1
      link(say, f'C  {name} brings what it looks at', np.array_equal(ov, want),
           'copies the bits and the flags into its own block; NONE and the MODE fallback bring "missing"')
  mode = heads['mode']
  ov = mode.V @ mode.O
  ok = rows(mode.Q) == {S.at('one')} and rows(mode.K) == mode_rows and np.abs(ov).sum() == len(task.modes)
  ok &= all(ov[S.at(f'is_{m.flag}'), S.at(f'mode_{m.flag}')] == 1 for m in task.modes)
  ok &= r.W >= max(period(m) for m in task.modes)
  link(say, 'B,C mode head', ok, f'only MODE tokens answer, each copying its mode; a group is at most '
       f'{max(period(m) for m in task.modes)} tokens and the window {r.W}, so every window holds one, all of the same mode')
  fallback = [t for t, name in enumerate(r.tokens) if (r.table[t] @ np.array([h.K[:, P] for h in r.heads[1:]]).T).any()]
  link(say, 'B  only MODE tokens are the fallback', sorted(r.tokens[t] for t in fallback) == sorted(m.flag.upper() for m in task.modes),
       'the 0.5 a query gives when the symbol it wants is not there goes to MODE tokens, which read as missing')
  bound = (r.W - 1) * np.exp(-r.sharp * 0.5)
  say(f'  bound  D  attention leaks at most {bound:.1e} of the weight')
  say(f'         {r.W - 1} other tokens in the window x e^-{r.sharp * 0.5:g}, whatever the input\'s length')
  return bound


def prove_cases(r, say, limit=1 << 20):
  task, S, mlp = r.task, r.S, r.mlp
  rows = lambda W: set(np.flatnonzero(np.abs(W).sum(1)))
  block = lambda name: set(range(S[name].start, S[name].stop))
  head_blocks = set().union(*(block(n) for n in S.blocks if '.' in n))
  modes = {S.at(f'mode_{m.flag}') for m in task.modes}
  outputs = block('out_bits') | {S.at('out_end')} | ({S.at('out_pad')} if 'out_pad' in S.blocks else set())
  ok = rows(mlp.W_in) <= head_blocks | modes | block('phase')
  ok &= set(np.flatnonzero(np.abs(mlp.W_out).sum(0))) <= outputs and rows(r.U) <= outputs | modes
  link(say, 'E  what the MLP and output layer can see', ok,
       'the MLP reads only fetched blocks, the mode and the phase, and writes only the output blocks; the '
       'output layer reads only those and the mode')
  ok = True
  for u, label in enumerate(mlp.units):
    m = task.mode(label.split(' place ')[0])
    place = int(label.split(' place ')[1].split(':')[0])
    want = np.zeros(r.P)
    if m.group[1] > 1:
      want[query_slot(m, place)::period(m)] = 1
    ok &= np.array_equal(mlp.W_in[S['phase'], u], want)
  link(say, 'E  each unit reads its place, nothing more', ok,
       'weight 1 on exactly the phases where its place is predicted (phase mod the group length); none when '
       'a group has one output')
  U_rows = np.flatnonzero(r.U.any(1))
  total, gap = 0, np.inf
  for m in task.modes:
    sym = {s.kind: s for s in task.symbols}
    out_names = sym[m.outputs].names
    seen = set()                                          # a fetch from the next group reads as missing (S)
    for place, kinds in reachable(task, m):
      kinds = tuple('missing' if r.distances[m.name][(place, i)] is None else k for i, k in enumerate(kinds))
      if (place, kinds) in seen:
        continue
      seen.add((place, kinds))
      ranges = [range(len(sym[m.outputs if m.reads_output(i) else m.inputs].names)) if k == 'symbol' else [0]
                for i, k in enumerate(kinds)]
      cases = list(itertools.product(*ranges))
      if len(cases) > limit:
        raise NotProved(f'{m.name} place {place}: {len(cases):,} cases is more than {limit:,}')
      for c0 in range(0, len(cases), 4096):
        chunk = np.array(cases[c0:c0 + 4096])
        noise = np.random.default_rng(c0)
        for phase in range(query_slot(m, place), r.P, period(m)):
          x = np.zeros((len(chunk), S.size))
          x[:, S.at('one')] = 1
          x[:, S.at(f'mode_{m.flag}')] = 1
          x[:, S.at('phase', phase)] = 1
          for i, (label, kind) in enumerate(zip(m.fetch, kinds)):
            name = f'{m.flag}_{label}'
            for b in range(task.value_bits):
              x[:, S.at(f'{name}.bits', b)] = (chunk[:, i] >> b) & 1
            if kind != 'symbol':
              x[:, S.at(f'{name}.{kind}')] = 1
          for other in task.modes:                         # the other modes' heads: anything
            if other is not m:
              for label in other.fetch:
                x[:, S[f'{other.flag}_{label}.bits']] = noise.integers(0, 2, (len(chunk), task.value_bits))
          y, _ = mlp(x)
          scores = y[:, U_rows] @ r.U[U_rows] + r.c
          top2 = np.sort(scores, 1)[:, -2:]
          says = rule_says(m, place, kinds)
          if says == 'end':
            want = np.full(len(chunk), r.id['END'])
          elif says == 'pad':
            want = np.full(len(chunk), r.id[m.pad])
          else:
            want = np.array([r.id[out_names[int(m.combine(*row, place))]] for row in chunk.tolist()])
          if not (scores.argmax(1) == want).all():
            link(say, 'E  MLP + output layer, every case', False, f'{m.name} place {place} kinds {kinds}')
          gap = min(gap, (top2[:, 1] - top2[:, 0]).min())
          total += len(chunk)
  link(say, 'E  MLP + output layer, every case', gap >= 2,
       f'{total:,} cases (every place at every phase it falls on, every kind of input a valid input can put '
       f'there, every value): the rules\' token wins every time, by at least {gap:g}')


def prove_ending(r, say):
  """G needs END before the runner stops feeding NONE groups: the rules end within 8 of them."""
  for m in r.task.modes:
    after = -(-(m.extra + 1) // m.group[1])               # groups of outputs written once the input is gone
    link(say, f'G  {m.name} ends in time', after <= 8,
         f'past the input\'s groups the rules write at most {m.extra} output{"s" if m.extra != 1 else ""} and END: {after} NONE '
         f'group{"s" if after > 1 else ""}, the runner feeds 8')


def prove(r, say=print, rules=True):
  if rules:
    prove_rules(r.task, say)
  prove_stream(r, say)
  bound = prove_attention(r, say)
  prove_cases(r, say)
  L = movement(r)
  link(say, 'F  the leak cannot change a token', bound < 1 / L,
       f'a score moves at most {L:g} x delta, so any delta under {1 / L:.2g} keeps the winner; the leak is {bound:.1e}')
  prove_ending(r, say)
  say('  holds  G  the whole answer')
  say('         by induction over the stream: the window before each output holds what S says, so B-F give')
  say('         the rules\' token, which the runner appends; the input\'s length appears nowhere')


if __name__ == '__main__':
  import tasks
  from qkvc.build import build_model
  task = tasks.find(sys.argv[1])
  print(f'proof that the {task.name} model writes exactly what its rules say,')
  print('for every valid input of any length; the rules are checked against its reference (link A)\n')
  try:
    prove(build_model(task))
    print('PROVED')
  except NotProved as e:
    sys.exit(f'not proved: {e}')
