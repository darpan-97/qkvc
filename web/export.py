"""The builder's base64 model for the web page.

    python web/export.py        -> web/weights.json, web/known_answers.json

The page runs the model live in the browser on exactly these numbers. Only the non-zero weights are
written, row by row, so the browser multiplies a vector by a matrix by visiting the vector's non-zero
entries: the same arithmetic as numpy, without the zeros.

known_answers.json holds what the Python model writes, and every score it gives at every step, for a
set of inputs. web/check_engine.js runs the browser engine on the same inputs and must get the same.
"""
import base64
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from qkvc.run import load                                   # noqa: E402


def by_row(W):
  """{row: [[column, value], ...]} for every non-zero entry."""
  rows = {}
  for r, col in zip(*np.nonzero(W)):
    rows.setdefault(int(r), []).append([int(col), float(W[r, col])])
  return rows


def export_weights(r):
  return {
      'size': r.S.size,
      'blocks': {name: list(span) for name, span in r.S.blocks.items()},
      'tokens': r.tokens,
      'window': r.W, 'phases': r.P, 'sharp': r.sharp,
      'modes': {m.name: {'flag': m.flag, 'group': list(m.group), 'fetch': list(m.fetch),
                         'distances': [[r.distances[m.name][(j, i)] for i in range(len(m.fetch))]
                                       for j in range(m.group[1])]} for m in r.task.modes},
      'token_table': by_row(r.table),
      'heads': [{'name': h.name, 'Q': by_row(h.Q), 'K': by_row(h.K), 'V': by_row(h.V), 'O': by_row(h.O),
                 'value_size': h.V.shape[1]} for h in r.heads],
      'mlp': {'W_in': by_row(r.mlp.W_in), 'b_in': r.mlp.b_in.tolist(), 'W_out': by_row(r.mlp.W_out),
              'units': r.mlp.units},
      'unembed': {'U': by_row(r.U), 'c': r.c.tolist()},
  }


def export_answers(model):
  """Inputs, the tokens the Python model writes, and its scores at every step."""
  rng = random.Random(3)
  cases = [('encode', b'Man'), ('decode', 'TWFu'), ('encode', b'Ma'), ('decode', 'TWE='), ('encode', b''),
           ('encode', b'Hello Darpan!'), ('decode', 'SGVsbG8gRGFycGFuIQ=='), ('encode', b'Type something here!')]
  cases += [('encode', bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40)))) for _ in range(6)]
  cases += [('decode', base64.b64encode(bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40)))).decode())
            for _ in range(6)]
  out = []
  for mode, data in cases:
    steps = list(model.steps(mode, data, detail=True))
    out.append({'mode': mode, 'input': list(data) if mode == 'encode' else data,
                'tokens': [s['token'] for s in steps],
                'scores': [np.round(s['scores'], 9).tolist() for s in steps]})
  return out


if __name__ == '__main__':
  model = load('base64')
  for name, data in (('weights.json', export_weights(model.r)), ('known_answers.json', export_answers(model))):
    path = os.path.join(HERE, name)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
      json.dump(data, f, separators=(',', ':'))
    print(f'{name}: {os.path.getsize(path):,} bytes')
