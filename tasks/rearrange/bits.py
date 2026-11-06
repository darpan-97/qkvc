"""Within each byte: reverse its 8 bits, or swap its two 4-bit halves."""
from qkvc.task import Mode, Rule, Symbols, Task

BYTES = tuple(f'x{v:02x}' for v in range(256))


def random_bytes(rng, length):
  return bytes(rng.randrange(256) for _ in range(length))


def per_byte(name, flag, rule):
  return Mode(name, flag, inputs='byte', outputs='byte', group=(1, 1), fetch=('byte',), sources=lambda n: (n,),
              combine=lambda byte, place: rule(byte), read=lambda data: [BYTES[b] for b in data],
              write=lambda names: bytes(int(n[1:], 16) for n in names), sample=random_bytes,
              rules=(Rule(0, 'byte', 'missing', 'end', 'no more bytes -> END'),))


def reverse(b):
  return sum(((b >> i) & 1) << (7 - i) for i in range(8))


def nibbles(b):
  return ((b >> 4) | (b << 4)) & 255


def reference(mode, data):
  rule = reverse if mode == 'reverse' else nibbles
  return [BYTES[rule(b)] for b in data]


TASK = Task(
    name='bits',
    symbols=(Symbols('byte', BYTES, 8),),
    flag_tokens={},
    modes=(per_byte('reverse', 'rev', reverse), per_byte('nibbles', 'nib', nibbles)),
    reference=reference,
)
