"""What the notebooks share: the repository on the import path, and small tools for looking inside a built model."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from qkvc.run import load                                   # noqa: E402

TEXT = b'Hello Darpan!'


def shown(name):
  """a token as it reads: x48 is x48 'H'"""
  if len(name) == 3 and name[0] == 'x':
    v = int(name[1:], 16)
    return f'{name} {chr(v)!r}' if 32 <= v < 127 else name
  return name


def stream(model, mode='encode', data=TEXT):
  """every token of the stream, group by group, ending with the group that writes END"""
  m = model.r.task.mode(mode)
  body = m.read(data)
  outs = [s['name'] for s in model.steps(mode, data)]
  size_in, size_out = m.group
  tokens, g = [], 0
  while True:
    chunk = body[g * size_in:(g + 1) * size_in]
    tokens += [m.flag.upper()] + chunk + ['NONE'] * (size_in - len(chunk))
    written = outs[g * size_out:(g + 1) * size_out]
    tokens += written
    if 'END' in written:
      return tokens
    g += 1


def row(model, name, t):
  """the 76 numbers of token `name` at position t of the stream"""
  r = model.r
  x = r.table[r.id[name]].copy()
  x[r.S.at('phase', t % r.P)] = 1
  return x


def rows(model, names):
  return np.array([row(model, n, t) for t, n in enumerate(names)])


def named(model, x):
  """the numbers of x that are not 0, by name"""
  names = model.r.S.names()
  return {names[i]: round(float(x[i]), 6) for i in np.flatnonzero(np.abs(x) > 1e-9)}


def attend(model, X, head):
  """head reading the last row of X: its scores, weights and what it adds. Attention sees the last W tokens only."""
  X = X[-model.r.W:]
  q = X[-1] @ head.Q
  s = model.r.sharp * (X @ head.K) @ q
  w = np.exp(s - s.max())
  w /= w.sum()
  return s, w, w @ (X @ head.V) @ head.O


def forward(model, X):
  """the last row of X through attention and the MLP: after attention, the hidden units, after the MLP"""
  x = X[-1].copy()
  for h in model.r.heads:
    x = x + attend(model, X, h)[2]
  after, hidden = model.r.mlp(x[None])
  return x, hidden[0], after[0]
