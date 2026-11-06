# %% [markdown]
# # 8. Proving it
#
# **The question:** one input worked. How do we know the others do?
#
# Testing cannot cover every input. The proof (`qkvc/verify.py`) reads the finished weights and checks a chain of links.
# If each link holds, the model writes what the task's rules say at every step of every valid input.
#
# | Link | Says |
# |---|---|
# | A | the task's rules match Python's base64: every full group, and sampled whole inputs |
# | S | the stream puts each symbol where the heads look |
# | B | each head looks where the distance table says |
# | C | each head brings what it looks at, and nothing else |
# | E | the MLP gives the right bits on every case a valid input can produce |
# | F | what attention leaks onto other tokens cannot change the winner |
# | G | together, by induction over the stream: the whole answer is right |
#
# ## 1. The proof of the base64 model

# %%
import contextlib
import dataclasses
import io

import helpers
from helpers import load
from qkvc.build import build_model
from qkvc.verify import NotProved, prove
import tasks

task = tasks.find('base64')
built = build_model(task)
prove(built)
print('PROVED')

# %% [markdown]
# Link A goes through all 16,777,216 three-byte groups and all 16,777,216 four-character groups, which is why this takes
# about a minute. The other links read the weights.
#
# ## 2. What is proved, and what is sampled
#
# The proof has two halves, and they do not have the same reach.
#
# | Claim | Links | Coverage |
# |---|---|---|
# | The model does exactly what the task's rules say | S, B to G | every valid input of any length, read from the weights |
# | The rules are the real algorithm | A | see below |
#
# For link A, base64 checks every full group. Inputs with a padded ending are sampled: 800 for decoding, 600 for encoding.
# Groups are independent, so for encoding the endings are just the inputs of 1 and 2 bytes, and there are few enough to
# check them all:

# %%
from qkvc.verify import rules_answer

m = task.mode('encode')
wrong = checked = 0
for length in (1, 2):
  for value in range(256 ** length):
    data = value.to_bytes(length, 'big')
    got, _ = rules_answer(task, 'encode', m.read(data))
    wrong += got != task.reference('encode', data)
    checked += 1
print(f'every 1- and 2-byte input: {checked:,} checked, {wrong} wrong')

# %% [markdown]
# So for base64 encoding, link A is complete. Decoding's padded endings stay sampled here. For tasks with a carry, such as
# addition, link A checks every short input and samples longer ones, because a carry runs along the whole input.
#
# ## 3. Does the proof catch mistakes?
#
# A proof that always passes is worthless. So we break the model on purpose, as `tests/controls.py` does, and check
# that the proof refuses it. A helper that returns the link that failed:

# %%
def rejected(built):
  try:
    with contextlib.redirect_stdout(io.StringIO()):
      prove(built, rules=False)
    return 'PASSED'
  except NotProved as e:
    return 'rejected at link ' + str(e)[:90]

print(rejected(build_model(task)))

# %% [markdown]
# A head that looks one token too far back, at one phase only:

# %%
c = build_model(task)
h = next(h for h in c.heads if h.name == 'dec_second')
d = c.distances['decode'][(1, 1)]
h.Q[c.S.at('phase', 5)] = 0
h.Q[c.S.at('phase', 5), (5 - d - 1) % c.P] = 1
print(rejected(c))

# %% [markdown]
# Two MLP wires swapped:

# %%
c = build_model(task)
units = [i for i, u in enumerate(c.mlp.units) if u.startswith('encode place 2:')][:2]
c.mlp.W_out[units] = c.mlp.W_out[units[::-1]]
print(rejected(c))

# %% [markdown]
# Attention made 250 times less sharp, so it leaks onto other tokens:

# %%
c = build_model(task)
c.sharp = 0.2
print(rejected(c))

# %% [markdown]
# `tests/controls.py` plants 11 such mistakes, in the weights and in the task files. The proof rejects all 11.
#
# ## 4. Why a mistake cannot hide
#
# A test finds a wrong wire only if a test input happens to use it. The proof reads every wire. And the two swapped wires
# above would be missed by any input where both bits are equal.
