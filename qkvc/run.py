"""Run a built task over a whole input, of any length.

    python -m qkvc run TASK MODE INPUT        e.g.  python -m qkvc run hex encode "Man"

The runner feeds each group's inputs (a group short of inputs is filled with NONE, and NONE groups
follow the last one until the model writes END), the model writes the group's outputs, and only the
last W tokens are kept.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qkvc.build import build_model                         # noqa: E402


class Model:
  """Runs a built model over a whole input, keeping only the last W tokens."""

  def __init__(self, r):
    self.r = r
    self.U_rows = np.flatnonzero(r.U.any(1))
    self.phase = np.array([r.S.at('phase', p) for p in range(r.P)])
    self.back = np.arange(r.W - 1, -1, -1)

  def steps(self, mode, data, detail=False):
    """Each output token as it is computed, END last. detail: also what each head read (`reads`:
    head name -> token, how far back), the MLP units that fired (`units`) and every token's score
    (`scores`), for `python -m qkvc trace` and web/export.py."""
    r, m = self.r, self.r.task.mode(mode)
    names = m.read(data)
    header, body = names[:m.header], names[m.header:]
    size_in, size_out = m.group
    keys = [[] for _ in r.heads]
    values = [[] for _ in r.heads]
    state = {'t': 0, 'row': None}
    window = []                                  # the tokens in the attention window

    def push(token):
      x = r.table[token].copy()
      x[r.S.at('phase', state['t'] % r.P)] = 1
      for k, h in enumerate(r.heads):
        keys[k].append(x @ h.K)
        values[k].append(x @ h.V)
        if len(keys[k]) > r.W:
          keys[k].pop(0), values[k].pop(0)
      window.append(r.tokens[token])
      del window[:-r.W]
      state['t'] += 1
      state['row'] = x

    groups = -(-len(body) // size_in)
    for g in range(groups + 8):
      chunk = body[g * size_in:(g + 1) * size_in]
      for name in [m.flag.upper()] + header + chunk + ['NONE'] * (size_in - len(chunk)):
        push(r.id[name])
      for _ in range(size_out):
        q = state['row']
        x = q.copy()
        reads = {}
        for k, h in enumerate(r.heads):
          s = r.sharp * (q @ h.Q) @ np.array(keys[k]).T
          w = np.exp(s - s.max())
          x += (w / w.sum()) @ np.array(values[k]) @ h.O
          reads[h.name] = (window[int(w.argmax())], len(window) - 1 - int(w.argmax()))
        x, hidden = r.mlp(x[None])
        scores = (x[:, self.U_rows] @ r.U[self.U_rows] + r.c)[0]
        token = int(scores.argmax())
        top = np.sort(scores)[-2:]
        step = {'token': token, 'name': r.tokens[token], 'gap': float(top[1] - top[0])}
        if detail:
          step.update(reads=reads, units=[u for u, v in zip(r.mlp.units, hidden[0]) if v > 0.5], scores=scores)
        yield step
        if r.tokens[token] == 'END':
          return
        push(token)
    raise RuntimeError(f'{mode}: no END after {groups + 8} groups')

  def fast(self, mode, data, chunk=4096):
    """The same outputs as steps(), many at once (Jacobi decoding: Song et al. 2021, "Accelerating
    Feedforward Computation via Parallel Nonlinear Equation Solving"; Santilli et al. 2023,
    "Accelerating Transformer Inference for Translation via Parallel Decoding").

    Every output is guessed, then a whole chunk of them is computed in one pass from the guesses.
    The first output of the chunk reads only settled tokens, so it is right; each one after it is
    right as long as every output before it came out the same as its guess. So everything up to and
    including the first output that changed is settled, and the next pass starts after it. When a
    pass changes nothing, every output in the chunk is the model's own choice given the tokens
    before it, which is exactly what steps() writes. Groups that do not read earlier outputs settle
    in two passes; a chain (carry, parity, checksum) settles about an output a pass, so the chunk
    shrinks for it."""
    r, m = self.r, self.r.task.mode(mode)
    names = m.read(data)
    header, body = names[:m.header], names[m.header:]
    size_in, size_out = m.group
    groups = -(-len(body) // size_in) + 8
    fed_len = 1 + len(header) + size_in
    L = fed_len + size_out
    tokens = np.full(groups * L, r.id['NONE'])
    for g in range(groups):
      chunk_in = body[g * size_in:(g + 1) * size_in]
      fed = [m.flag.upper()] + header + chunk_in + ['NONE'] * (size_in - len(chunk_in))
      tokens[g * L:g * L + fed_len] = [r.id[n] for n in fed]
    slots = (np.arange(groups)[:, None] * L + fed_len + np.arange(size_out)).ravel()   # where outputs go
    end, out, gaps = r.id['END'], [], []
    done, size = 0, min(chunk, len(slots))
    while done < len(slots):
      at, first = slots[done:done + size], True
      while len(at):
        new, gap = self._pass(tokens, at - 1)
        changed = np.flatnonzero(new != tokens[at])
        settled = changed[0] + 1 if len(changed) else len(at)
        tokens[at] = new
        stop = np.flatnonzero(new[:settled] == end)
        if len(stop):
          settled = stop[0] + 1
        out += new[:settled].tolist()
        gaps += gap[:settled].tolist()
        done += settled
        if len(stop):
          return [{'token': t, 'name': r.tokens[t], 'gap': g} for t, g in zip(out, gaps)]
        at = at[settled:]
        # the first pass over a chunk starts from blind guesses; a later one that still settles
        # under half of what is left is a chain, so the chunk halves
        if not first and 2 * settled < settled + len(at):
          size = max(size_out, size // 2)
          break
        first = False
      else:
        size = min(chunk, size * 2)
    raise RuntimeError(f'{mode}: no END after {groups} groups')

  def _pass(self, tokens, queries):
    """The model at positions `queries` of tokens, all at once: each one's top token and its gap."""
    r = self.r
    lo, hi = max(0, queries[0] - r.W + 1), queries[-1] + 1
    x = r.table[tokens[lo:hi]]                              # fancy indexing: already a copy
    x[np.arange(hi - lo), self.phase[np.arange(lo, hi) % r.P]] = 1
    window = queries[:, None] - self.back                   # the last W tokens, oldest first
    valid = window >= 0
    local = np.where(valid, window, queries[:, None]) - lo
    xq = x[queries - lo]
    y = xq.copy()
    for h in r.heads:
      k, v = x @ h.K, x @ h.V
      s = r.sharp * (k[local] @ (xq @ h.Q)[:, :, None])[:, :, 0]
      s[~valid] = -np.inf
      w = np.exp(s - s.max(1, keepdims=True))
      w /= w.sum(1, keepdims=True)
      y += (w[:, None, :] @ v[local])[:, 0] @ h.O
    y, _ = r.mlp(y)
    scores = y[:, self.U_rows] @ r.U[self.U_rows] + r.c
    top = np.sort(scores, 1)[:, -2:]
    return scores.argmax(1), top[:, 1] - top[:, 0]

  def generate(self, mode, data):
    return [s['name'] for s in self.fast(mode, data) if s['name'] != 'END']


def load(name):
  import tasks
  return Model(build_model(tasks.find(name)))


if __name__ == '__main__':
  name, mode, text = sys.argv[1:4]
  model = load(name)
  data = text.encode() if model.r.task.mode(mode).inputs == 'byte' else text
  print(' '.join(model.generate(mode, data)))
