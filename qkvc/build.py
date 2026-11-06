"""qkvc, the builder: a task (task.py) in, a transformer's weights out.

The stream is written group by group, each group's inputs then its outputs:

    [MODE] [header] [the group's inputs] [the group's outputs]   [MODE] [header] [inputs] [outputs]  ...

Laid out like this, every symbol an output reads is a fixed distance back from the token before it,
the same in every group. The model has

    one attention layer   a head per mode that reads which mode this is, and a head per fetched
                          symbol that looks a fixed distance back
    one MLP layer         units that make each output bit from the fetched bits, read off combine();
                          and one unit per rule, for END and padding
    an output layer       every token scored by how many of its bits agree with the output's, plus
                          a constant for its kind

A position is its phase, the token's index mod P, one-hot. A head that fetches a symbol d tokens back
asks for phase (its own phase - d) mod P, and every token's key is its phase. Attention is a sliding
window over the last W tokens, W <= P, so within the window every phase appears at most once and the
match is exact (sliding-window attention as in Longformer and Mistral). A symbol not there, before the
stream starts or not yet written, falls back to the group's MODE token, which every query half
matches and which reads as missing. Nothing depends on the input's length.
"""
import math
from dataclasses import dataclass

import numpy as np

SHARP = 100.0          # attention scores x 100: the named position wins by at least 50
BIG = 20.0             # the kind constant of the output layer: more than any bit score


class Layout:
  """The residual stream: every number has a name."""

  def __init__(self):
    self.size, self.blocks = 0, {}

  def add(self, name, width=1):
    self.blocks[name] = (self.size, width)
    self.size += width

  def __getitem__(self, name):
    start, width = self.blocks[name]
    return slice(start, start + width)

  def at(self, name, i=0):
    start, width = self.blocks[name]
    assert 0 <= i < width, (name, i)
    return start + i

  def names(self):
    """The name of every number, in order: 'bits[3]', 'is_sep', ..."""
    return [name if width == 1 else f'{name}[{i}]' for name, (_, width) in self.blocks.items() for i in range(width)]


class Head:
  def __init__(self, name, size, address, value):
    self.name = name
    self.Q, self.K = np.zeros((size, address)), np.zeros((size, address))
    self.V, self.O = np.zeros((size, value)), np.zeros((value, size))
    self.rows = {}

  def project(self, x, W):
    """x @ W, using only the rows of W that are not all zero and the inputs that are not 0."""
    if id(W) not in self.rows:
      self.rows[id(W)] = np.flatnonzero(W.any(1))
    rows = self.rows[id(W)]
    rows = rows[x[:, rows].any(0)]
    return x[:, rows] @ W[rows]

  def __call__(self, x):
    """What the head adds at each position of x (causal), and its attention weights."""
    scores = SHARP * self.project(x, self.Q) @ self.project(x, self.K).T
    scores[np.triu_indices(len(x), 1)] = -np.inf
    weights = np.exp(scores - scores.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    return weights @ self.project(x, self.V) @ self.O, weights


class MLP:
  def __init__(self, W_in, b_in, W_out, units):
    self.W_in, self.b_in, self.W_out, self.units = W_in, b_in, W_out, units
    self.rows = np.flatnonzero(W_in.any(1))

  def __call__(self, x):
    hidden = np.maximum(0, x[:, self.rows] @ self.W_in[self.rows] + self.b_in)
    return x + hidden @ self.W_out, hidden


def wiring(combine, place, bits):
  """[(fetch index, input bit, output bit)]: where each fetched bit lands, found by setting one
  input bit at a time (bits: how many bits each fetched symbol has); None if a bit lands in two
  places or two bits in one."""
  wires = []
  for which, width in enumerate(bits):
    for k in range(width):
      values = [0] * len(bits)
      values[which] = 1 << k
      landed = int(combine(*values, place))
      if landed & (landed - 1):
        return None
      if landed:
        wires.append((which, k, landed.bit_length() - 1))
  if len({o for _, _, o in wires}) != len(wires):
    return None
  return wires


def every_value(sizes):
  """Arrays of every combination of fetched values (each below its size), one array per fetched symbol."""
  grid = np.meshgrid(*[np.arange(size) for size in sizes], indexing='ij')
  return [g.ravel() for g in grid]


def bit_functions(combine, place, bits, sizes, out_bits, max_support=4):
  """For each output bit: the input bits it depends on (found by flipping each input bit across every
  combination of values) and the patterns of them that make it 1. None if an output bit depends on
  more than max_support input bits, or is not a function of the bits found."""
  values = every_value(sizes)
  result = lambda vals: np.asarray(combine(*vals, place)) * np.ones_like(vals[0])
  base = result(values)
  found = []
  for o in range(out_bits):
    f = (base >> o) & 1
    support = []
    for i, width in enumerate(bits):
      for k in range(width):
        flipped = list(values)
        flipped[i] = values[i] ^ (1 << k)
        valid = flipped[i] < sizes[i]                      # a flipped value must still be a symbol
        if valid.any() and (f[valid] != (result(flipped)[valid] >> o) & 1).any():
          support.append((i, k))
    if len(support) > max_support:
      return None
    pattern = sum(((values[i] >> k) & 1) << j for j, (i, k) in enumerate(support)) if support else values[0] * 0
    seen = {}
    for p, v in zip(pattern.tolist(), f.tolist()):
      if seen.setdefault(p, v) != v:
        return None
    found.append((o, support, sorted(p for p, v in seen.items() if v)))
  return found


def wiring_is_exact(wires, combine, place, sizes):
  """Whether the wires give combine's value for every combination of fetched values."""
  values = every_value(sizes)
  wired = sum(((values[which] >> k) & 1) << o for which, k, o in wires) if wires else values[0] * 0
  return np.array_equal(wired, np.asarray(combine(*values, place)) * np.ones_like(values[0]))


def build_mlp(task, S, symbols, place_input, table_limit=1 << 12):
  """AND units: ReLU(weighted inputs that are each 0 or 1, minus a bias) that is 1 only when every
  input it wants is 1 and every input it wants 0 is 0. Each reads its mode's flag and 'this output is
  at place r' (place_input(m, r): the phases where place r is written; none when a group has one
  output). And either

    one fetched bit, and writes it to one output bit: bit wiring, when combine() only moves bits
    a few fetched bits, matching one pattern of them that makes an output bit 1: bit functions, when
      every output bit depends on at most 4 input bits (XOR with a key: 2 units a bit)
    every bit of the fetched inputs, matching one combination of their values exactly, and writes
    that combination's output value: a lookup table (Tracr's categorical MLP)

  choosing wiring when it is exact, otherwise whichever of the other two needs fewer units,

  and one unit per rule, for END and padding."""
  cols, biases, outs, labels = [], [], [], []

  def unit(weights, m, r, writes, label):
    """weights: {(slot name, index): weight}; the bias makes the unit 1 only at an exact match. The
    place input (place_input(m, r): slots that add up to 1 exactly at place r) counts as one input;
    with one output per group there is none."""
    w = np.zeros(S.size)
    for slot, weight in weights.items():
      w[S.at(*slot)] += weight
    place = place_input(m, r)
    for slot in place:
      w[S.at(*slot)] += 1
    out = np.zeros(S.size)
    for slot in writes:
      out[S.at(*slot)] = 1
    wanted = sum(1 for weight in weights.values() if weight > 0) + bool(place)
    cols.append(w), biases.append(-(wanted - 1)), outs.append(out), labels.append(label)

  for m in task.modes:
    size, out_sym = m.group[1], symbols[m.outputs]
    alphabets = [symbols[m.outputs if m.reads_output(i) else m.inputs] for i in range(len(m.fetch))]
    bits, sizes = [a.bits for a in alphabets], [len(a.names) for a in alphabets]
    for place in range(size):
      wires = wiring(m.combine, place, bits)
      if wires is not None and wiring_is_exact(wires, m.combine, place, sizes):
        for which, k, o in wires:
          label = m.fetch[which]
          unit({(f'mode_{m.flag}', 0): 1, (f'{m.flag}_{label}.bits', k): 1}, m, place, [('out_bits', o)],
               f'{m.name} place {place}: {label}.bit{k} -> out bit {o}')
        continue
      combos = int(np.prod(sizes))
      functions = bit_functions(m.combine, place, bits, sizes, out_sym.bits) if combos <= 1 << 16 else None
      if functions is not None and sum(len(ones) for _, _, ones in functions) <= combos:
        for o, support, ones in functions:
          for p in ones:
            weights = {(f'mode_{m.flag}', 0): 1}
            for j, (i, k) in enumerate(support):
              weights[(f'{m.flag}_{m.fetch[i]}.bits', k)] = 1 if (p >> j) & 1 else -1
            shown = ' '.join(f'{m.fetch[i]}.bit{k}={(p >> j) & 1}' for j, (i, k) in enumerate(support)) or 'always'
            unit(weights, m, place, [('out_bits', o)], f'{m.name} place {place}: {shown} -> out bit {o}')
        continue
      values = every_value(sizes)
      if len(values[0]) > table_limit:
        raise ValueError(f'{m.name} place {place}: combine() does not just move bits, and a table would '
                         f'need {len(values[0]):,} units (at most {table_limit:,})')
      results = np.asarray(m.combine(*values, place)) * np.ones_like(values[0])
      for combo, result in zip(zip(*values), results):
        weights = {(f'mode_{m.flag}', 0): 1}
        for label, value, alphabet in zip(m.fetch, combo, alphabets):
          for k in range(alphabet.bits):
            weights[(f'{m.flag}_{label}.bits', k)] = 1 if (value >> k) & 1 else -1
        shown = ', '.join(f'{label} {a.names[v]}' for label, v, a in zip(m.fetch, combo, alphabets))
        unit(weights, m, place, [('out_bits', b) for b in range(out_sym.bits) if (int(result) >> b) & 1],
             f'{m.name} place {place}: {shown} -> {out_sym.names[int(result)]}')
  for m in task.modes:
    for rule in m.rules:
      unit({(f'mode_{m.flag}', 0): 1, **{(f'{m.flag}_{head}.{flag}', 0): 1 for head, flag in rule.conditions}},
           m, rule.place,
           [('out_end', 0) if rule.writes == 'end' else ('out_pad', 0)], f'{m.name} place {rule.place}: {rule.text}')
  return MLP(np.array(cols).T, np.array(biases, dtype=float), np.array(outs), labels)


def build_output_layer(task, S, symbols, tokens):
  """Each candidate scores +1 for every output bit that agrees with its own and -1 for every one
  that does not, plus BIG for its kind in the current mode; '=' wins when the pad flag is on, END
  (3 x BIG) when the end flag is on. Tokens nothing writes score -BIG."""
  U, c = np.zeros((S.size, len(tokens))), np.zeros(len(tokens))
  for t, name in enumerate(tokens):
    written = False
    for s in task.symbols:
      if name not in s.names:
        continue
      for m in task.modes:
        if m.outputs != s.kind:
          continue
        if not written:
          value = s.names.index(name)
          for b in range(s.bits):
            sign = 1 if (value >> b) & 1 else -1
            U[S.at('out_bits', b), t] += 2 * sign
            c[t] -= sign
        written = True
        U[S.at(f'mode_{m.flag}'), t] += BIG
        if m.pad:
          U[S.at('out_pad'), t] -= BIG
    for m in task.modes:
      if m.pad and name == m.pad:
        written = True
        U[S.at(f'mode_{m.flag}'), t] += BIG
        U[S.at('out_pad'), t] += BIG
    if name == 'END':
      written = True
      U[S.at('out_end'), t] += 3 * BIG
    if not written:
      c[t] = -BIG
  return U, c


def period(m):
  """Tokens per group: MODE, the header, the inputs, the outputs."""
  return 1 + m.header + m.group[0] + m.group[1]


def query_slot(m, j):
  """The slot (index within the group) of the token before output j: the token that predicts it."""
  return m.header + m.group[0] + j


def stream_index(m, n, i, g):
  """Where in the stream fetch i of output n (in group g) reads."""
  p = m.sources(n)[i]
  T, h, (size_in, size_out) = period(m), m.header, m.group
  if m.reads_output(i):
    return (p // size_out) * T + 1 + h + size_in + p % size_out
  if 0 <= p < h:
    return g * T + 1 + p
  return ((p - h) // size_in) * T + 1 + h + (p - h) % size_in


def distances(m, deep=6):
  """{(place, fetch index): how far back from the token predicting it}; None for a symbol not yet in
  the stream when it is predicted (it reads as missing). The same in every group, or an error."""
  T, size_out = period(m), m.group[1]
  out = {}
  for j in range(size_out):
    for i in range(len(m.fetch)):
      found = {g * T + query_slot(m, j) - stream_index(m, g * size_out + j, i, g) for g in (deep, deep + 1)}
      if len(found) != 1:
        raise ValueError(f'{m.name}: fetch {m.fetch[i]} at place {j} is not the same distance back in every group')
      d = found.pop()
      out[(j, i)] = d if d >= 0 else None
  return out


@dataclass
class Built:
  task: object
  tokens: list
  id: dict
  S: Layout
  table: np.ndarray
  heads: list
  mlp: object
  U: np.ndarray
  c: np.ndarray
  P: int                  # phases
  W: int                  # attention window
  distances: dict         # mode name -> {(place, fetch index): distance back}
  sharp: float = SHARP


def build_model(task):
  symbols = {s.kind: s for s in task.symbols}
  bits = task.value_bits
  dist = {m.name: distances(m) for m in task.modes}
  lcm = math.lcm(*(period(m) for m in task.modes))
  window = max([period(m) for m in task.modes] + [d + 1 for ds in dist.values() for d in ds.values() if d is not None])
  P = lcm * -(-window // lcm)                               # a multiple of every period, at least the window

  tokens = [n for s in task.symbols for n in s.names] + list(task.flag_tokens.values())
  tokens += [m.flag.upper() for m in task.modes] + ['NONE', 'END']
  ids = {t: i for i, t in enumerate(tokens)}
  S = Layout()
  S.add('one')
  S.add('bits', bits)
  for kind in [s.kind for s in task.symbols] + list(task.flag_tokens) + ['none', 'end'] + [m.flag for m in task.modes]:
    S.add(f'is_{kind}')
  S.add('phase', P)
  for m in task.modes:
    S.add(f'mode_{m.flag}')
  for m in task.modes:
    for label in m.fetch:
      S.add(f'{m.flag}_{label}.bits', bits)
      for kind in task.flag_tokens:
        S.add(f'{m.flag}_{label}.{kind}')
      S.add(f'{m.flag}_{label}.missing')
  S.add('out_bits', bits)
  if any(m.pad for m in task.modes):
    S.add('out_pad')
  S.add('out_end')

  table = np.zeros((len(tokens), S.size))
  special = {'NONE': 'none', 'END': 'end'} | {m.flag.upper(): m.flag for m in task.modes}
  for t, name in enumerate(tokens):
    table[t, S.at('one')] = 1
    for s in task.symbols:
      if name in s.names:
        table[t, S.at(f'is_{s.kind}')] = 1
        for b in range(bits):
          table[t, S.at('bits', b)] = (s.names.index(name) >> b) & 1
    for kind, token in task.flag_tokens.items():
      if name == token:
        table[t, S.at(f'is_{kind}')] = 1
    if name in special:
      table[t, S.at(f'is_{special[name]}')] = 1

  mode_flags = [S.at(f'is_{m.flag}') for m in task.modes]
  heads = [Head('mode', S.size, 1, len(task.modes))]       # every position finds a MODE token
  heads[0].Q[S.at('one'), 0] = 1
  for i, m in enumerate(task.modes):
    heads[0].K[mode_flags[i], 0] = 1
    heads[0].V[mode_flags[i], i] = 1
    heads[0].O[i, S.at(f'mode_{m.flag}')] = 1
  flags = list(task.flag_tokens)
  for m in task.modes:
    T = period(m)
    for i, label in enumerate(m.fetch):
      name = f'{m.flag}_{label}'
      h = Head(name, S.size, P + 1, bits + len(flags) + 1)
      for j in range(m.group[1]):
        d = dist[m.name][(j, i)]
        if d is None:
          continue
        for phase in range(query_slot(m, j), P, T):          # every phase where output j is predicted
          h.Q[S.at('phase', phase), (phase - d) % P] = 1
      h.Q[S.at('one'), P] = 0.5
      for phase in range(P):
        h.K[S.at('phase', phase), phase] = 1
      for f in mode_flags:
        h.K[f, P] = 1
      for b in range(bits):
        h.V[S.at('bits', b), b] = 1
        h.O[b, S.at(f'{name}.bits', b)] = 1
      for k, kind in enumerate(flags):
        h.V[S.at(f'is_{kind}'), bits + k] = 1
        h.O[bits + k, S.at(f'{name}.{kind}')] = 1
      for f in [S.at('is_none')] + mode_flags:               # NONE, or the fallback: missing
        h.V[f, bits + len(flags)] = 1
      h.O[bits + len(flags), S.at(f'{name}.missing')] = 1
      heads.append(h)

  def place_input(m, r):
    if m.group[1] == 1:
      return []
    return [('phase', phase) for phase in range(query_slot(m, r), P, period(m))]

  mlp = build_mlp(task, S, symbols, place_input)
  U, c = build_output_layer(task, S, symbols, tokens)
  return Built(task, tokens, ids, S, table, heads, mlp, U, c, P, window, dist)
