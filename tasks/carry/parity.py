"""Running parity of a bit string: output n is the XOR of bits 0 to n, so the last output is the
parity of the whole string.

Parity is the standard example of a function transformers find hard to learn when they must answer
in one step (every bit matters); written out one step at a time, each step is one XOR with the
previous output.  python -m qkvc run parity parity 1101001
"""
import itertools
import operator

from qkvc.task import Mode, Rule, Symbols, Task


def reference(mode, data):
  return [str(p) for p in itertools.accumulate((int(b) for b in data), operator.xor)]


TASK = Task(
    name='parity',
    symbols=(Symbols('bit', ('0', '1'), 1),),
    flag_tokens={},
    modes=(Mode('parity', 'par', inputs='bit', outputs='bit', group=(1, 1), fetch=('bit', 'prev'),
                from_outputs=('prev',), sources=lambda n: (n, n - 1), combine=lambda bit, prev, place: bit ^ prev,
                read=list, write=''.join, sample=lambda rng, length: ''.join(rng.choice('01') for _ in range(length)),
                rules=(Rule(0, 'bit', 'missing', 'end', 'no more bits -> END'),)),),
    reference=reference,
)
