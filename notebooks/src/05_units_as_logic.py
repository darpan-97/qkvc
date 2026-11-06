# %% [markdown]
# # 5. MLP units as logic
#
# **The question:** attention brought the bytes. How does the MLP turn them into the output's bits?
#
# Each unit of the MLP is a ReLU acting as an AND gate.

# %%
import numpy as np

import helpers
from helpers import forward, load, named, rows, stream

model = load('base64')
r = model.r
mlp = r.mlp
tokens = stream(model)
print('units:', len(mlp.units))
for u in mlp.units:
  if u.startswith('encode place 0'):
    print(' ', u)

# %% [markdown]
# ## 1. One unit
#
# `encode place 0: first.bit6 -> out bit 4` means: for the first output of a group, copy bit 6 of the first byte to
# output bit 4. Its weights:

# %%
names = r.S.names()
u = mlp.units.index('encode place 0: first.bit6 -> out bit 4')
reads = {names[i]: float(mlp.W_in[i, u]) for i in np.flatnonzero(mlp.W_in[:, u])}
writes = [names[i] for i in np.flatnonzero(mlp.W_out[u])]
print('reads: ', reads)
print('bias:  ', mlp.b_in[u])
print('writes:', writes)

# %% [markdown]
# `phase[3]` is the token that predicts the first output. The unit is ReLU(a + b + c - 2). With inputs of 0 or 1 that
# is 1 only when all three are 1:

# %%
for a in (0, 1):
  for b in (0, 1):
    for c in (0, 1):
      print(a, b, c, '->', max(0, a + b + c - 2))

# %% [markdown]
# ## 2. How the builder finds the wires
#
# The builder never reads `encode_char`'s code. It calls it with one input bit set at a time and sees where the bit lands.

# %%
from qkvc.build import wiring
from tasks.rearrange.base64 import encode_char

print('first output is first >> 2:')
for which, k, o in wiring(encode_char, 0, [8, 8]):
  print(f'  {("first", "second")[which]} bit {k} -> out bit {o}')
print('combine with only bit 6 of first set:', encode_char(1 << 6, 0, 0), '= bit 4')

# %% [markdown]
# ## 3. Which units fire for `S`
#
# The vector from chapter 4 goes through the MLP. `H` has bits 3 and 6 on, so two units fire.

# %%
X = rows(model, tokens[:4])
before, hidden, after = forward(model, X)
for unit, value in zip(mlp.units, hidden):
  if value > 0.5:
    print(f'{value:.0f}  {unit}')

# %%
print({k: v for k, v in named(model, after).items() if k.startswith('out')})

# %% [markdown]
# Output bits 1 and 4 are `010010`, which is 18: the character `S`.
#
# ## 4. All four outputs of the first group

# %%
for n, step in enumerate(model.steps('encode', helpers.TEXT, detail=True)):
  if n == 4:
    break
  print(f"output {n + 1}: {step['name']}")
  for unit in step['units']:
    print('   ', unit)

# %% [markdown]
# Encoding uses 24 wires: 6 bits for each of the 4 places. No unit computes anything; each one copies a bit to its place.
# Base64 is pure wiring. Chapter 9 shows algorithms that need more.
