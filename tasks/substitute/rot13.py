"""ROT13: every letter moved 13 places round the alphabet; doing it twice gives the text back."""
import codecs

from qkvc.task import Task
from tasks.substitute import BYTES, SYMBOLS, per_byte, shift_letters


def reference(mode, data):
  return [BYTES[b] for b in codecs.encode(data.decode('latin-1'), 'rot13').encode('latin-1')]


TASK = Task(name='rot13', symbols=SYMBOLS, flag_tokens={},
            modes=(per_byte('rot13', 'r13', lambda b: shift_letters(b, 13)),), reference=reference)
