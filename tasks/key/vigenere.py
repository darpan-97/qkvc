"""Vigenere's cipher on the letters A-Z: every message letter moved on (encrypt) or back (decrypt) by
the key letter at its position (A = 0, B = 1, ...), the key repeating.

Input: a 4-letter key, then the message.  python -m qkvc run vigenere encrypt LEMONATTACKATDAWN
"""
import string

from qkvc.task import Symbols, Task
from tasks.key import keyed

LETTERS = string.ascii_uppercase


def sample(rng, length):
  return ''.join(rng.choice(LETTERS) for _ in range(4 + length))


def reference(mode, data):
  """Each key letter's own shifted alphabet, as Vigenere's table (tabula recta) has it."""
  key, message = data[:4], data[4:]
  rows = {k: LETTERS[LETTERS.index(k):] + LETTERS[:LETTERS.index(k)] for k in key}
  if mode == 'encrypt':
    return [rows[key[i % 4]][LETTERS.index(c)] for i, c in enumerate(message)]
  return [LETTERS[rows[key[i % 4]].index(c)] for i, c in enumerate(message)]


TASK = Task(
    name='vigenere',
    symbols=(Symbols('letter', tuple(LETTERS), 5),),
    flag_tokens={},
    modes=(keyed('encrypt', 'enc', 'letter', 4, lambda m, k, place: (m + k) % 26, sample),
           keyed('decrypt', 'dec', 'letter', 4, lambda m, k, place: (m - k) % 26, sample)),
    reference=reference,
)
