# %% [markdown]
# # 3. Named numbers
#
# **The question:** what does a token look like inside the model?
#
# In a trained transformer a token is a vector of numbers nobody chose. Here each of the 76 numbers has a name and
# one job.

# %%
import helpers
from helpers import load, named, row, stream

model = load('base64')
r = model.r
print(f"{'name':<22}{'numbers':>8}   meaning")
meaning = {
  'one': 'always 1', 'bits': "the token's value", 'is_char': 'a base64 character', 'is_byte': 'a byte', 'is_pad': 'the = sign',
  'is_none': 'NONE', 'is_end': 'END', 'is_dec': 'the DEC token', 'is_enc': 'the ENC token',
  'phase': 'position in the group of 8', 'mode_dec': 'decoding', 'mode_enc': 'encoding',
  'out_bits': 'the output being assembled', 'out_pad': 'write =', 'out_end': 'write END'}
for head in ('enc_first', 'enc_second', 'dec_first', 'dec_second'):
  meaning |= {f'{head}.bits': f'what head {head} fetched', f'{head}.pad': 'it fetched =', f'{head}.missing': 'it found nothing'}
for name, (start, width) in r.S.blocks.items():
  print(f'{name:<22}{width:>8}   {meaning[name]}')
print('total', r.S.size)

# %% [markdown]
# ## 1. A token going in
#
# The embedding table holds what a token is; the position adds its `phase`. Here are the numbers that are not 0 for
# the byte `H`, which is token 1 of the stream. `H` is `0x48`, binary `01001000`, so bits 3 and 6 are on.

# %%
tokens = stream(model)
print(tokens[:4])
print(named(model, row(model, 'x48', 1)))

# %% [markdown]
# For the character `S`, value 18, binary `010010`, at token 4:

# %%
print(named(model, row(model, 'S', 4)))

# %% [markdown]
# And the mode token at the start of a group:

# %%
print(named(model, row(model, 'ENC', 0)))

# %% [markdown]
# Most of the 76 numbers are 0 when a token enters. The fetched and output blocks start empty: attention and the MLP
# write into them, as the next chapters show. Reading a vector means reading its names, which is why every step of
# this model can be checked.
