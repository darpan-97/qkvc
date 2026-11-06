"""Substitution: each output byte is a function of one input byte, any function. The builder builds
the MLP as a lookup table, one AND unit per input value (Tracr's categorical MLP)."""
from qkvc.task import Mode, Rule, Symbols

BYTES = tuple(f'x{v:02x}' for v in range(256))
SYMBOLS = (Symbols('byte', BYTES, 8),)


def per_byte(name, flag, function):
  """A mode that maps every byte through `function` (which must also work on numpy arrays)."""
  return Mode(name, flag, inputs='byte', outputs='byte', group=(1, 1), fetch=('byte',), sources=lambda n: (n,),
              combine=lambda byte, place: function(byte), read=lambda data: [BYTES[b] for b in data],
              write=lambda names: bytes(int(n[1:], 16) for n in names),
              sample=lambda rng, length: bytes(rng.randrange(256) for _ in range(length)),
              rules=(Rule(0, 'byte', 'missing', 'end', 'no more bytes -> END'),))


def shift_letters(b, k):
  """Letters moved k places round the alphabet, case kept; every other byte unchanged."""
  upper = ((b >= 65) & (b <= 90)) * 1
  lower = ((b >= 97) & (b <= 122)) * 1
  return upper * ((b - 65 + k) % 26 + 65) + lower * ((b - 97 + k) % 26 + 97) + (1 - upper - lower) * b
