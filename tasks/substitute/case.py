"""Letter case: ASCII letters to upper case or to lower case; every other byte unchanged."""
from qkvc.task import Task
from tasks.substitute import BYTES, SYMBOLS, per_byte


def upper(b):
  return b - 32 * ((b >= 97) & (b <= 122))


def lower(b):
  return b + 32 * ((b >= 65) & (b <= 90))


def reference(mode, data):
  return [BYTES[b] for b in (data.upper() if mode == 'upper' else data.lower())]


TASK = Task(name='case', symbols=SYMBOLS, flag_tokens={},
            modes=(per_byte('upper', 'up', upper), per_byte('lower', 'low', lower)), reference=reference)
