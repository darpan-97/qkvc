"""Keyed: each output combines a message symbol with a key symbol. The key comes first in the input, a
header every group reads; output n reads message symbol n and key symbol n mod (key length). The
builder builds the MLP as bit functions when each output bit depends on a few input bits (XOR), or
as a lookup table over (message, key) pairs (Vigenere)."""
from qkvc.task import Mode, Rule


def keyed(name, flag, symbols, key_length, combine, sample, read=list, write=''.join):
  """A mode whose input is `key_length` key symbols, then the message."""
  return Mode(name, flag, inputs=symbols, outputs=symbols, group=(key_length, key_length), fetch=('message', 'key'),
              sources=lambda n: (key_length + n, n % key_length), combine=combine, read=read, write=write,
              sample=sample, header=key_length,
              rules=tuple(Rule(j, 'message', 'missing', 'end', 'no more message -> END') for j in range(key_length)))
