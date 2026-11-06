"""Binary: every byte as 8 digits 0/1, the highest bit first.

A byte reads 8 digits, so decoding fetches 8 inputs. Writing 8 outputs per byte multiplies the output
positions, so this task takes inputs of up to 256 symbols.
"""
from qkvc.task import Mode, Rule, Symbols, Task

BYTES = tuple(f'x{v:02x}' for v in range(256))
DIGITS = tuple(f'd{i}' for i in range(8))                  # fetch labels: digit 0 is the highest bit


def random_bytes(rng, length):
  return bytes(rng.randrange(256) for _ in range(length))


def reference(mode, data):
  if mode == 'decode':
    return [BYTES[int(data[i:i + 8], 2)] for i in range(0, len(data), 8)]
  return list(''.join(f'{b:08b}' for b in data))


TASK = Task(
    name='binary',
    symbols=(Symbols('digit', ('0', '1'), 1), Symbols('byte', BYTES, 8)),
    flag_tokens={},
    modes=(
        Mode('decode', 'dec', inputs='digit', outputs='byte', group=(8, 1), fetch=DIGITS,
             sources=lambda n: tuple(8 * n + i for i in range(8)),
             combine=lambda *digits: sum(d << (7 - i) for i, d in enumerate(digits[:8])),
             read=list, write=''.join,
             sample=lambda rng, length: ''.join(f'{b:08b}' for b in random_bytes(rng, length)),
             rules=(Rule(0, 'd0', 'missing', 'end', 'no more digits -> END'),)),
        Mode('encode', 'enc', inputs='byte', outputs='digit', group=(1, 8), fetch=('byte',),
             sources=lambda n: (n // 8,), combine=lambda byte, place: (byte >> (7 - place)) & 1,
             read=lambda data: [BYTES[b] for b in data], write=lambda names: bytes(int(n[1:], 16) for n in names),
             sample=random_bytes,
             rules=(Rule(0, 'byte', 'missing', 'end', 'no more bytes -> END'),)),
    ),
    reference=reference,
)
