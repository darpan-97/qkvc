"""Every task in tasks/: build it, check its answers against its reference (inputs up to 20,000
symbols), count its parameters, and (with --prove) prove it.

    python tests/tasks.py [--prove]
"""
import contextlib
import io
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tasks                                                # noqa: E402
from qkvc.build import build_model                         # noqa: E402
from qkvc.run import Model                                  # noqa: E402
from qkvc.verify import NotProved, prove                    # noqa: E402

LONG = (5_000, 20_000)

prove_too = '--prove' in sys.argv
print('| Task | Modes | Right answers | Longest input | Parameters | Non-zero |' + (' Proof |' if prove_too else ''))
print('|---|---|---:|---:|---:|---:|' + ('---|' if prove_too else ''))
failed = False
for family, task in tasks.every_task():
  c = build_model(task)
  model, rng, right, total, longest, same = Model(c), random.Random(0), 0, 0, 0, True
  for m in task.modes:
    for length in list(range(13)) + [rng.randrange(13, 200) for _ in range(10)] + list(LONG):
      data = m.sample(rng, length)
      total += 1
      if model.generate(m.name, data) == task.reference(m.name, data):
        right += 1
        longest = max(longest, len(m.read(data)))
      if length < 200:                                      # the fast runner against the step-by-step one
        same &= [s['token'] for s in model.fast(m.name, data)] == [s['token'] for s in model.steps(m.name, data)]
  arrays = [c.table, c.mlp.W_in, c.mlp.b_in, c.mlp.W_out, c.U, c.c] + [w for h in c.heads for w in (h.Q, h.K, h.V, h.O)]
  stored = sum(a.size for a in arrays) + c.P * c.S.size     # + a one-hot row per phase
  nonzero = sum(int(np.count_nonzero(a)) for a in arrays) + c.P
  line = f'| {task.name} | {", ".join(m.name for m in task.modes)} | {right} of {total} | {longest:,} | {stored:,} | {nonzero:,} |'
  if prove_too:
    try:
      with contextlib.redirect_stdout(io.StringIO()):
        prove(c)
      line += ' proved |'
    except NotProved as e:
      line += f' not proved: {e} |'
      failed = True
  if not same:
    line += ' fast runner differs from steps() |'
  print(line, flush=True)
  failed |= right != total or not same
sys.exit(1 if failed else 0)
