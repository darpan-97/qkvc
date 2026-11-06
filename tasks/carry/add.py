"""Binary addition of two numbers of the same length, written least significant digit first (the
order the carry runs), one digit longer than the numbers for the final carry.

Input "a+b" with a and b binary, most significant digit first: python -m qkvc run add add 1011+0110

The carry into digit n is not written as a token: it is worked out from digit n - 1's two inputs and
the sum digit the model wrote there, c = sum ^ a ^ b, and the carry out of that digit is the
majority of a, b and c.
"""
from qkvc.task import Mode, Rule, Symbols, Task


def read(data):
  """'a+b' -> a0 b0 a1 b1 ..., least significant digits first."""
  a, b = data.split('+')
  if len(a) != len(b):
    raise ValueError('the two numbers must have the same number of digits')
  return [d for pair in zip(a[::-1], b[::-1]) for d in pair]


def write(names):
  if len(names) % 2:
    raise ValueError('an odd number of digits is not two numbers')
  return ''.join(names[0::2])[::-1] + '+' + ''.join(names[1::2])[::-1]


def add_digit(a, b, a_prev, b_prev, sum_prev, place):
  carry_prev = sum_prev ^ a_prev ^ b_prev                   # the carry into the previous digit
  carry = (a_prev & b_prev) | (a_prev & carry_prev) | (b_prev & carry_prev)
  return a ^ b ^ carry


def sample(rng, length):
  digits = lambda: ''.join(rng.choice('01') for _ in range(length))
  return digits() + '+' + digits()


def reference(mode, data):
  a, b = data.split('+')
  if not a:
    return []
  return list(format(int(a, 2) + int(b, 2), f'0{len(a) + 1}b')[::-1])


TASK = Task(
    name='add',
    symbols=(Symbols('digit', ('0', '1'), 1),),
    flag_tokens={},
    modes=(Mode('add', 'add', inputs='digit', outputs='digit', group=(2, 1),
                fetch=('a', 'b', 'a_prev', 'b_prev', 'sum_prev'), from_outputs=('sum_prev',),
                sources=lambda n: (2 * n, 2 * n + 1, 2 * n - 2, 2 * n - 1, n - 1), combine=add_digit,
                read=read, write=write, sample=sample, extra=1,
                rules=(Rule(0, 'a', 'missing', 'end', 'no digits here or before -> END',
                            also=(('a_prev', 'missing'),)),)),),
    reference=reference,
)
