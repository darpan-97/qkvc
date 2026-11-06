"""Byte order (endianness): reverse the bytes of every 16-bit or 32-bit word. Inputs and outputs are
both bytes; each output copies one input from elsewhere in its word."""
from qkvc.task import Mode, Rule, Symbols, Task

BYTES = tuple(f'x{v:02x}' for v in range(256))


def word_swap(size):
  def sample(rng, length):
    return bytes(rng.randrange(256) for _ in range(size * length))
  return Mode(f'swap{8 * size}', f's{8 * size}', inputs='byte', outputs='byte', group=(size, size), fetch=('byte',),
              sources=lambda n: (size * (n // size) + size - 1 - n % size,), combine=lambda byte, place: byte,
              read=lambda data: [BYTES[b] for b in data], write=lambda names: bytes(int(n[1:], 16) for n in names),
              sample=sample, rules=(Rule(0, 'byte', 'missing', 'end', 'no more words -> END'),))


def reference(mode, data):
  size = int(mode[4:]) // 8
  return [BYTES[b] for i in range(0, len(data), size) for b in reversed(data[i:i + size])]


TASK = Task(
    name='byteswap',
    symbols=(Symbols('byte', BYTES, 8),),
    flag_tokens={},
    modes=(word_swap(2), word_swap(4)),
    reference=reference,
)
