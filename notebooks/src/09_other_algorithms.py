# %% [markdown]
# # 9. Other algorithms
#
# **The question:** base64 only moves bits. What does the builder do for algorithms that compute something?
#
# It uses the cheapest of three forms for the MLP:
#
# | Form | Used when | Example |
# |---|---|---|
# | Wiring | each input bit lands in one output bit | hex, base64 |
# | Bit functions | each output bit depends on at most 4 input bits | XOR |
# | Lookup table | otherwise, up to 4,096 units per output position | ROT13 |

# %%
import helpers
from helpers import load
import tasks

for family, task in tasks.every_task():
  print(f'{family:<11}{task.name:<11}', ', '.join(m.name for m in task.modes))

# %% [markdown]
# ## 1. Wiring: hex
#
# One byte in, two digits out. Each digit is 4 bits of the byte.

# %%
hexm = load('hex')
print(' '.join(hexm.generate('encode', b'Hi')))
for u in hexm.r.mlp.units:
  if u.startswith('encode'):
    print(u)

# %% [markdown]
# ## 2. Bit functions: XOR with a key
#
# The output bit is `message bit XOR key bit`. No wire can do that, because the output depends on two bits. The builder
# finds which bits each output depends on, by flipping them, and writes one unit per pattern that gives a 1.

# %%
xor = load('xor')
print(len(xor.r.mlp.units), 'units')
for u in [u for u in xor.r.mlp.units if u.startswith('xor1 place 0') and 'out bit 0' in u]:
  print(u)
print()
print(' '.join(xor.generate('xor1', b'Khello')), '  key K')

# %% [markdown]
# Two units per output bit: bit 0 of the output is 1 when the message and key bits differ.
#
# ## 3. Lookup table: ROT13
#
# Each byte maps to another, with no pattern in the bits. So the MLP is a table: one unit per byte value.

# %%
rot = load('rot13')
print(len(rot.r.mlp.units), 'units')
for u in rot.r.mlp.units[65:68]:
  print(u)
print(''.join(chr(int(t[1:], 16)) for t in rot.generate('rot13', b'Hello Darpan!')))

# %% [markdown]
# ## 4. State: addition
#
# Addition needs a carry, and the model has no memory. It reads its own previous output instead: the carry into a digit can
# be worked out from the previous digit's inputs and the sum digit the model wrote there.

# %%
add = load('add')
m = add.r.task.mode('add')
print('reads:', m.fetch, ' of which earlier outputs:', m.from_outputs)
print('1011 + 0110 =', ' '.join(add.generate('add', '1011+0110')), '  (least significant digit first: 10001 = 17)')

# %% [markdown]
# ## 5. All 16
#
# Every task builds from a file of a few lines. Every one is proved by `python tests/tasks.py --prove`.

# %%
for family, task in tasks.every_task():
  r = load(task.name).r
  print(f'{task.name:<11}{len(r.heads):>3} heads {len(r.mlp.units):>5} units')
