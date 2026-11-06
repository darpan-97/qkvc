# %% [markdown]
# # 10. The limits
#
# **The question:** what can this model shape not do?
#
# It has one attention layer and one MLP layer, and its heads look a fixed distance back. It fits algorithms that work like
# a sliding window with a small carried state. Here the builder refuses algorithms that do not fit.

# %%
import dataclasses

import helpers
from qkvc.build import build_model
import tasks

rot13 = tasks.find('rot13')
base = rot13.modes[0]


def try_build(mode, task=rot13):
  try:
    built = build_model(dataclasses.replace(task, modes=(mode,)))
    return f'built: {len(built.mlp.units)} units'
  except ValueError as e:
    return f'refused: {e}'

print(try_build(base))

# %% [markdown]
# ## 1. The distance must be fixed: reversing a string
#
# To reverse a 1,000-byte string, output `n` reads input `999 - n`. The distance back grows with every output.

# %%
print(try_build(dataclasses.replace(base, sources=lambda n: (999 - n,))))

# %% [markdown]
# A distance that grows any other way is refused too:

# %%
print(try_build(dataclasses.replace(base, sources=lambda n: (n * n,))))

# %% [markdown]
# Reversing inside a fixed block is fine, because the distance repeats. That is the byte swap task:

# %%
from helpers import load

swap = load('byteswap')
print(' '.join(swap.generate('swap32', bytes([1, 2, 3, 4, 5, 6, 7, 8]))))

# %% [markdown]
# ## 2. The table has a limit: multiplying two bytes
#
# A function of two whole bytes that does not just move bits needs a unit per combination: 65,536.

# %%
bytes_task = tasks.find('xor')
mode = next(m for m in bytes_task.modes if m.name == 'xor1')
times = dataclasses.replace(mode, combine=lambda message, key, place: (message * key) & 255)
print(try_build(times, bytes_task))

# %% [markdown]
# ## 3. Out of reach, not attempted
#
# These follow from the limits above; they were not tried.
#
# | Algorithm | Why |
# |---|---|
# | Sorting | where a value goes depends on every value |
# | Search, LZ77 | where to read depends on the data |
# | Brackets of any depth, counting | the state grows with the input |
# | Long multiplication | each digit depends on many digits |
# | SHA-256, AES | many rounds of mixing |
#
# ## 4. Whose limit is it?
#
# These are limits of this design, not of transformers. They are also what makes the proof possible: with fixed positions and
# few bits per output, every case can be checked. More layers would reach further, and the proof would get harder.
