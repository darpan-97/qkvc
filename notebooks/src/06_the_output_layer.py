# %% [markdown]
# # 6. The output layer
#
# **The question:** the MLP left the bits `010010` in `out_bits`. How does that become the token `S`?
#
# The output layer gives each of the 325 tokens a score, and the highest score is written.

# %%
import numpy as np

import helpers
from helpers import forward, load, named, rows, stream

model = load('base64')
r = model.r
tokens = stream(model)
X = rows(model, tokens[:4])
before, hidden, after = forward(model, X)
print('out_bits:', [int(after[r.S.at('out_bits', b)]) for b in range(6)], '(bit 0 first)')

# %% [markdown]
# ## 1. The scores

# %%
scores = after @ r.U + r.c
order = np.argsort(-scores)
for t in order[:7]:
  print(f'{r.tokens[t]:<6} {scores[t]:6.1f}')
print('...')
print('lowest:', r.tokens[order[-1]], scores[order[-1]])

# %% [markdown]
# `S` wins. A byte token such as `x41` cannot win here: the score has a constant for the token's kind, and in encode mode
# only characters get it.
#
# ## 2. Why `S` wins
#
# Each character scores +1 for each of its 6 bits that matches `out_bits` and -1 for each that does not. `S` matches
# all 6. The runners-up differ in one bit, so they match 5 and miss 1: 4 from their bits, which is 2 behind. Every
# character also has the kind constant of 20, so the scores above are 26 and 24.

# %%
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
target = ALPHABET.index('S')
for name in ['S'] + [r.tokens[t] for t in order[1:4]]:
  value = ALPHABET.index(name)
  agree = 6 - bin(value ^ target).count('1')
  print(f'{name}  {value:06b}  agrees on {agree} of 6, so {agree - (6 - agree):+d} from its bits, score {scores[r.id[name]]:.1f}')

# %% [markdown]
# The margin is what the proof relies on: a wrong token is always at least 2 behind, so a small leak in attention cannot
# change the answer. Chapter 8 proves the leak is far smaller than that.
#
# ## 3. Every output of the text

# %%
for n, step in enumerate(model.steps('encode', helpers.TEXT)):
  print(step['name'], f"(won by {step['gap']:.0f})", end='   ')
  if n % 4 == 3:
    print()
