# %% [markdown]
# # 2. The stream
#
# **The question:** a transformer sees only a sequence of tokens. How can one fixed set of weights handle input of
# any length?
#
# By laying the tokens out so that what an output needs is always the same distance back.

# %%
import random
import base64

import helpers
from helpers import TEXT, load, shown, stream

model = load('base64')
r = model.r
tokens = stream(model)
for g in range(0, len(tokens), 8):
  print(f'group {g // 8 + 1}:', ' '.join(f'{t:>5}' for t in tokens[g:g + 8]))

# %% [markdown]
# Each group is `ENC`, 3 bytes, then the 4 characters the model writes. `NONE` fills a group that ran out of bytes,
# and a last group writes `END`. (Tokens `x48` and so on are bytes: `x48` is `H`.)
#
# ## 1. The token before an output predicts it
#
# A transformer predicts the next token from the current one. So the token that predicts `S` is the `l` before it,
# and the token that predicts `G` is `S`.
#
# ## 2. Distances
#
# The builder works out how far back each output's bytes are from the token that predicts it.

# %%
print('period (tokens per group):', r.P)
print(f"{'output':<8}{'first':>7}{'second':>8}")
d = r.distances['encode']
for place in range(4):
  print(f'{place + 1:<8}{d[(place, 0)]!s:>7}{d[(place, 1)]!s:>8}')

# %% [markdown]
# `None` means this output does not need a second byte. The same table holds in every group. Checking that for the
# first two groups, by counting positions in the stream:

# %%
for group_start in (0, 8):
  for place in range(4):
    predictor = group_start + 3 + place
    source = group_start + 1 + (0, 0, 1, 2)[place]
    print(f'group at {group_start}, output {place + 1}: predicted at token {predictor}, first byte at token {source}, '
          f'{predictor - source} back')

# %% [markdown]
# ## 3. Why this gives any length
#
# The model looks at the last 8 tokens only. Nothing in it depends on where in the stream it is, except the position
# within the group, which repeats. So a long input is the same work repeated.

# %%
print('attention window:', r.W, 'tokens')
rng = random.Random(0)
for length in (1, 13, 1000, 100000):
  data = bytes(rng.randrange(256) for _ in range(length))
  same = ''.join(model.generate('encode', data)) == base64.b64encode(data).decode()
  print(f'{length:>7} bytes: same as Python: {same}')

# %% [markdown]
# Next: what is inside one of these tokens.
