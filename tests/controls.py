"""The proof (qkvc/verify.py) must reject a model or a task with a planted mistake, at the link
meant to catch it.

    python tests/controls.py
"""
import contextlib
import dataclasses
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qkvc.build import build_model                         # noqa: E402
from qkvc.verify import NotProved, prove                    # noqa: E402
import tasks                                                # noqa: E402
from tasks.rearrange.base64 import decode_byte              # noqa: E402


def rejected(built, rules=False):
  """The link that failed, or None if the proof held."""
  try:
    with contextlib.redirect_stdout(io.StringIO()):
      prove(built, rules=rules)
    return None
  except NotProved as e:
    return str(e)


base64, add, rot13 = tasks.find('base64'), tasks.find('add'), tasks.find('rot13')
assert rejected(build_model(base64)) is None, 'the proof fails on the real model'

planted = []

c = build_model(base64)                                # a head one token off, at one phase
h = next(h for h in c.heads if h.name == 'dec_second')
d = c.distances['decode'][(1, 1)]
phase = 5                                                   # decode's place 1 is predicted at slot 5
h.Q[c.S.at('phase', phase)] = 0
h.Q[c.S.at('phase', phase), (phase - d - 1) % c.P] = 1
planted.append(('dec_second looks one token too far back, at one phase', c, False))

c = build_model(base64)                                # the window cut below the farthest fetch
c.W = max(d for ds in c.distances.values() for d in ds.values() if d is not None)
planted.append(('a window too short for the farthest fetch', c, False))

c = build_model(base64)                                # attention far less sharp
c.sharp = 0.2
planted.append(('attention 250x less sharp, so it leaks', c, False))

c = build_model(base64)                                # two MLP wires swapped
units = [i for i, u in enumerate(c.mlp.units) if u.startswith('encode place 2:')][:2]
c.mlp.W_out[units] = c.mlp.W_out[units[::-1]]
planted.append(('two MLP wires swapped at encode place 2', c, False))

c = build_model(base64)                                # a unit that reads one phase too many
u = next(i for i, name in enumerate(c.mlp.units) if name.startswith('encode place 1:'))
c.mlp.W_in[c.S.at('phase', 0), u] = 1
planted.append(('an MLP unit that also fires at a phase of another place', c, False))

c = build_model(rot13)                                 # a lookup-table unit that writes the wrong letter
unit = c.mlp.units.index('rot13 place 0: byte x41 -> x4e')
c.mlp.W_out[unit] = 0
c.mlp.W_out[unit, [c.S.at('out_bits', b) for b in range(8) if (0x4f >> b) & 1]] = 1
planted.append(("rot13's lookup-table unit for 'A' writes 'O' instead of 'N'", c, False))

c = build_model(add)                                   # the carry head reads two outputs back
h = next(h for h in c.heads if h.name == 'add_sum_prev')
d = c.distances['add'][(0, 4)]
h.Q[c.S['phase']] = 0
h.Q[c.S.at('phase', 2), (2 - d - 4) % c.P] = 1
planted.append(("add's carry head reads the output two groups back, not one", c, False))

c = build_model(add)                                   # the fallback token answers like a symbol
for h in c.heads[1:]:
  h.V[c.S.at('is_add'), 0] = 1
planted.append(('a MODE fallback that brings bit 0 as well as "missing"', c, False))

def swapped(a, b, place):                                   # still pure wiring, so it builds
  v = decode_byte(a, b, place)
  return ((v & 1) << 1) | ((v >> 1) & 1) | (v & 252) if place == 1 else v


modes = tuple(dataclasses.replace(m, combine=swapped) if m.name == 'decode' else m for m in base64.modes)
c = build_model(dataclasses.replace(base64, modes=modes))  # a task whose rules disagree with its reference
planted.append(("a task whose decode rule swaps two output bits, so it disagrees with Python's base64", c, True))

or_carry = lambda a, b, ap, bp, sp, place: a ^ b ^ (ap | bp)  # the carry as OR, not majority
c = build_model(dataclasses.replace(add, modes=(dataclasses.replace(add.modes[0], combine=or_carry),)))
planted.append(('an addition task whose carry is "a or b" instead of the majority of a, b and the carry in', c, True))

square = dataclasses.replace(rot13.modes[0], sources=lambda n: (n * n,))
c = dataclasses.replace(build_model(rot13), task=dataclasses.replace(rot13, modes=(square,)))
planted.append(('a task whose output n reads input n * n: no fixed distance', c, False))

for what, built, rules in planted:
  failed = rejected(built, rules)
  assert failed, f'the proof passed a planted mistake: {what}'
  print(f'rejected at link {failed.split()[0]:<4} {what}')
