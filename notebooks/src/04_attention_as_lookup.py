# %% [markdown]
# # 4. Attention as an exact lookup
#
# **The question:** to write `S`, the model needs the byte `H`, 2 tokens back. How does attention find exactly that
# token?
#
# An attention head compares the current token's **query** with every token's **key**. The weights here make the query
# name a position and the key state a position, so only one token matches.

# %%
import numpy as np

import helpers
from helpers import attend, load, named, rows, shown, stream

model = load('base64')
r = model.r
tokens = stream(model)
X = rows(model, tokens[:4])              # ENC H e l : the model is about to write S
print([shown(t) for t in tokens[:4]])
head = next(h for h in r.heads if h.name == 'enc_first')
print('Q', head.Q.shape, 'K', head.K.shape, 'V', head.V.shape, 'O', head.O.shape)

# %% [markdown]
# ## 1. The query
#
# The token predicting `S` is `l`, at phase 3. The distance table says `first` is 2 back, so at phase 3 the head asks for
# phase 1. That is one entry in Q:

# %%
q = X[-1] @ head.Q
print('query (one number per phase, then a fallback):')
print(q)
print('Q row for phase[3]:', head.Q[r.S.at('phase', 3)])

# %% [markdown]
# The 1 is at index 1: "phase 1". The last number, 0.5, is a fallback explained below.
#
# ## 2. The keys
#
# Each token's key states its own phase. The mode token also matches the fallback.

# %%
keys = X @ head.K
for t, k in zip(tokens[:4], keys):
  print(f'{shown(t):<10} key {k}')

# %% [markdown]
# ## 3. Scores and weights
#
# The score is query times key, multiplied by 100 so that the winner takes all of the softmax.

# %%
scores, weights, added = attend(model, X, head)
for t, s, w in zip(tokens[:4], scores, weights):
  print(f'{shown(t):<10} score {s:6.1f}   weight {w:.20f}')

# %% [markdown]
# `H` scores 100, `ENC` scores 50 through the fallback, and the rest 0. The softmax puts all the weight on `H`. If no
# token were at the asked position, `ENC` would win, and that reads as "missing".
#
# ## 4. What it brings
#
# V and O copy the token's bits into the head's own named block.

# %%
print(named(model, added))

# %% [markdown]
# Bits 3 and 6 of `H` are now in `enc_first.bits`. The second head does the same for `e`, 1 back:

# %%
second = next(h for h in r.heads if h.name == 'enc_second')
s2, w2, added2 = attend(model, X, second)
print('reads', shown(tokens[int(np.argmax(w2))]))
print(named(model, added2))

# %% [markdown]
# `e` is `0x65`, binary `01100101`: bits 0, 2, 5 and 6.
#
# There is also a `mode` head, which finds the group's `ENC` token and sets `mode_enc`. Together:

# %%
x = X[-1].copy()
for h in r.heads:
  x = x + attend(model, X, h)[2]
print(named(model, x))

# %% [markdown]
# This vector is what the MLP receives: the token `l`, and on top of it what attention brought. The two decoding heads
# run too, find nothing to read, and set `dec_first.missing` and `dec_second.missing`. Nothing uses those while encoding,
# because every encoding unit also needs `mode_enc`.
