"""Atbash: the alphabet reversed (a <-> z, b <-> y, ...), case kept; it is its own inverse."""
import string

from qkvc.task import Task
from tasks.substitute import BYTES, SYMBOLS, per_byte

UP, LOW = string.ascii_uppercase, string.ascii_lowercase


def atbash(b):
  upper = ((b >= 65) & (b <= 90)) * 1
  lower = ((b >= 97) & (b <= 122)) * 1
  return upper * (155 - b) + lower * (219 - b) + (1 - upper - lower) * b       # 'A' + 'Z' = 155, 'a' + 'z' = 219


def reference(mode, data):
  table = str.maketrans(UP + LOW, UP[::-1] + LOW[::-1])
  return [BYTES[b] for b in data.decode('latin-1').translate(table).encode('latin-1')]


TASK = Task(name='atbash', symbols=SYMBOLS, flag_tokens={}, modes=(per_byte('atbash', 'atb', atbash),),
            reference=reference)
