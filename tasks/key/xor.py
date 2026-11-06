"""XOR with a repeating key, the usual way malware hides its strings: every message byte XOR the key
byte at its position, the key repeating. Doing it twice with the same key gives the message back.

Input: the key (1 or 4 bytes), then the message.  python -m qkvc run xor xor4 "KEY!hello"
"""
import itertools

from qkvc.task import Symbols, Task
from tasks.key import keyed

BYTES = tuple(f'x{v:02x}' for v in range(256))


def mode(key_length):
  return keyed(f'xor{key_length}', f'x{key_length}', 'byte', key_length, lambda message, key, place: message ^ key,
               sample=lambda rng, length: bytes(rng.randrange(256) for _ in range(key_length + length)),
               read=lambda data: [BYTES[b] for b in data], write=lambda names: bytes(int(n[1:], 16) for n in names))


def reference(mode, data):
  key_length = int(mode[3:])
  key, message = data[:key_length], data[key_length:]
  return [BYTES[m ^ k] for m, k in zip(message, itertools.cycle(key))]


TASK = Task(name='xor', symbols=(Symbols('byte', BYTES, 8),), flag_tokens={}, modes=(mode(1), mode(4)),
            reference=reference)
