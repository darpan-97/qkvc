"""Running XOR checksum of bytes: output n is the XOR of bytes 0 to n, so the last output is the
longitudinal redundancy check (LRC) of the whole input.  python -m qkvc run checksum xor "hello"
"""
import itertools
import operator

from qkvc.task import Mode, Rule, Symbols, Task

BYTES = tuple(f'x{v:02x}' for v in range(256))


def reference(mode, data):
  return [BYTES[v] for v in itertools.accumulate(data, operator.xor)]


TASK = Task(
    name='checksum',
    symbols=(Symbols('byte', BYTES, 8),),
    flag_tokens={},
    modes=(Mode('xor', 'cx', inputs='byte', outputs='byte', group=(1, 1), fetch=('byte', 'prev'),
                from_outputs=('prev',), sources=lambda n: (n, n - 1), combine=lambda byte, prev, place: byte ^ prev,
                read=lambda data: [BYTES[b] for b in data], write=lambda names: bytes(int(n[1:], 16) for n in names),
                sample=lambda rng, length: bytes(rng.randrange(256) for _ in range(length)),
                rules=(Rule(0, 'byte', 'missing', 'end', 'no more bytes -> END'),)),),
    reference=reference,
)
