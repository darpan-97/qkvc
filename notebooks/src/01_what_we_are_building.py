# %% [markdown]
# # 1. What we are building
#
# **The question:** can a transformer compute an algorithm exactly, with no training?
#
# qkvc takes a short description of an algorithm and places every weight of a transformer that computes it.
# These notebooks follow one input through one model, with real values at every step:
# the text `Hello Darpan!` through base64.
#
# | Chapter | Shows |
# |---|---|
# | 1 | base64 manually, and the task file |
# | 2 | the token stream, and why fixed distances work for any length |
# | 3 | the named numbers of a token |
# | 4 | attention as an exact lookup |
# | 5 | MLP units as logic gates |
# | 6 | the output layer |
# | 7 | padding and the end |
# | 8 | proving the model correct |
# | 9 | other algorithms |
# | 10 | the limits |

# %%
import base64
import inspect

import helpers
from helpers import TEXT, load

model = load('base64')
written = model.generate('encode', TEXT)
print('model: ', ''.join(written))
print('python:', base64.b64encode(TEXT).decode())

# %% [markdown]
# ## 1. Base64 manually
#
# Base64 takes 3 bytes, 24 bits, and cuts them into 4 pieces of 6 bits. Each piece picks one of 64 characters.

# %%
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
group = TEXT[:3]
bits = ''.join(f'{b:08b}' for b in group)
print('bytes:     ', ' '.join(f'{chr(b)!r:8}' for b in group))
print('bits:      ', ' '.join(f'{b:08b}' for b in group))
pieces = [bits[i:i + 6] for i in range(0, 24, 6)]
print('6-bit:     ', ' '.join(pieces))
print('values:    ', ' '.join(str(int(p, 2)) for p in pieces))
print('characters:', ' '.join(ALPHABET[int(p, 2)] for p in pieces))

# %% [markdown]
# `Hello Darpan!` is 13 bytes: 4 full groups and 1 byte left over.

# %%
for i in range(0, len(TEXT), 3):
  chunk = TEXT[i:i + 3]
  print(f'{chunk.decode()!r:7} -> {base64.b64encode(chunk).decode()}')

# %% [markdown]
# ## 2. The task file
#
# All the builder knows about base64 is in `tasks/rearrange/base64.py`. The important parts, for encoding:

# %%
from tasks.rearrange import base64 as task_file

print(inspect.getsource(task_file.encode_sources))
print(inspect.getsource(task_file.encode_char))

# %% [markdown]
# `encode_sources(n)` says which two bytes output `n` reads. `encode_char` says how to make the output from them.
# `place` is the output's position in its group of 4.
#
# ## 3. What the builder makes from it

# %%
r = model.r
print('tokens:              ', len(r.tokens))
print('numbers per token:   ', r.S.size)
print('attention heads:     ', [h.name for h in r.heads])
print('MLP units:           ', len(r.mlp.units))
arrays = [r.table, r.mlp.W_in, r.mlp.b_in, r.mlp.W_out, r.U, r.c] + [w for h in r.heads for w in (h.Q, h.K, h.V, h.O)]
print('parameters:          ', sum(a.size for a in arrays) + r.P * r.S.size)

# %% [markdown]
# The model has one attention layer, one MLP layer and an output layer: the parts of any transformer.
# Nothing was learned. The next chapters show where each of those numbers comes from.
