"""Hex: every byte as two digits 0-9 a-f, the high 4 bits first."""
from qkvc.task import Mode, Rule, Symbols, Task

DIGITS = tuple('0123456789abcdef')
BYTES = tuple(f'x{v:02x}' for v in range(256))


def random_bytes(rng, length):
  return bytes(rng.randrange(256) for _ in range(length))


def reference(mode, data):
  if mode == 'decode':
    return [BYTES[b] for b in bytes.fromhex(data)]
  return list(data.hex())


TASK = Task(
    name='hex',
    symbols=(Symbols('digit', DIGITS, 4), Symbols('byte', BYTES, 8)),
    flag_tokens={},
    modes=(
        Mode('decode', 'dec', inputs='digit', outputs='byte', group=(2, 1), fetch=('high', 'low'),
             sources=lambda n: (2 * n, 2 * n + 1), combine=lambda high, low, place: (high << 4) | low,
             read=list, write=''.join, sample=lambda rng, length: random_bytes(rng, length).hex(),
             rules=(Rule(0, 'high', 'missing', 'end', 'no more digits -> END'),)),
        Mode('encode', 'enc', inputs='byte', outputs='digit', group=(1, 2), fetch=('byte',),
             sources=lambda n: (n // 2,), combine=lambda byte, place: (byte >> 4) if place == 0 else (byte & 15),
             read=lambda data: [BYTES[b] for b in data], write=lambda names: bytes(int(n[1:], 16) for n in names),
             sample=random_bytes,
             rules=(Rule(0, 'byte', 'missing', 'end', 'no more bytes -> END'),)),
    ),
    reference=reference,
)
