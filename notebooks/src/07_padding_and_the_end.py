# %% [markdown]
# # 7. Padding and the end
#
# **The question:** `Hello Darpan!` is 13 bytes, so the last group has only `!`. How does the model write `IQ==` and
# then stop?

# %%
import helpers
from helpers import load, named, rows, shown, stream, forward, TEXT

model = load('base64')
r = model.r
tokens = stream(model)
print(' '.join(tokens[32:]))

# %% [markdown]
# A group that runs out of bytes is filled with `NONE`. After the last group comes one more, all `NONE`.
#
# ## 1. The rules in the task file
#
# The rules say what to do when a fetched byte is missing. Each becomes one MLP unit.

# %%
m = r.task.mode('encode')
for rule in m.rules:
  print(f'output {rule.place + 1} of a group, {rule.head} is {rule.flag}: {rule.text}')
print()
for u in r.mlp.units:
  if u.startswith('encode') and '.bit' not in u:
    print(u)

# %% [markdown]
# `NONE` reads as missing: the heads carry a `missing` flag instead of bits.
#
# ## 2. The last group, step by step

# %%
steps = list(model.steps('encode', TEXT, detail=True))
for n in range(16, 21):
  s = steps[n]
  print(f"output {n + 1}: {s['name']}  (won by {s['gap']:.0f})")
  for head, (token, back) in s['reads'].items():
    if head.startswith('enc_'):
      print(f'  {head:<11} reads {shown(token)}, {back} back')
  for unit in s['units']:
    print('  unit', unit)

# %% [markdown]
# - `I` is `!` shifted right by 2: `00100001` gives `001000`, which is 8.
# - `Q` takes the last 2 bits of `!` and 4 zero bits: `010000`, which is 16. The second byte is missing, so its wires
#   do not fire.
# - The next two outputs have no byte at all, so the rules write `=`.
# - The extra group finds nothing at all, so the rule writes `END`.
#
# ## 3. How a rule beats the bits
#
# A rule sets `out_pad` or `out_end`, and the output layer gives those tokens a large constant. That is why `=` and `END`
# win by 32 or more, where an ordinary character wins by 2.

# %%
X = rows(model, tokens[:38])             # up to the Q: the model is about to write output 19
before, hidden, after = forward(model, X)
print('at output 19:', {k: v for k, v in named(model, after).items() if k.startswith('out') or k.startswith('enc_first')})
scores = after @ r.U + r.c
top = scores.argsort()[::-1][:3]
print([(r.tokens[t], round(float(scores[t]), 1)) for t in top])

# %% [markdown]
# ## 4. The whole answer

# %%
import base64

answer = ''.join(s['name'] for s in steps if s['name'] != 'END')
print(answer)
print(base64.b64encode(TEXT).decode())
print(answer == base64.b64encode(TEXT).decode())
