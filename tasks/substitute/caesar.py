"""Caesar's cipher: every letter moved 3 places on to encrypt, 3 back to decrypt."""
import string

from qkvc.task import Task
from tasks.substitute import BYTES, SYMBOLS, per_byte, shift_letters

UP, LOW = string.ascii_uppercase, string.ascii_lowercase


def reference(mode, data):
  k = 3 if mode == 'encrypt' else -3
  table = str.maketrans(UP + LOW, UP[k:] + UP[:k] + LOW[k:] + LOW[:k])
  return [BYTES[b] for b in data.decode('latin-1').translate(table).encode('latin-1')]


TASK = Task(name='caesar', symbols=SYMBOLS, flag_tokens={},
            modes=(per_byte('encrypt', 'enc', lambda b: shift_letters(b, 3)),
                   per_byte('decrypt', 'dec', lambda b: shift_letters(b, -3))),
            reference=reference)
