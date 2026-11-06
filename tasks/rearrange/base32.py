"""base32 (RFC 4648): every 5 bytes as 8 characters of 5 bits, '=' padding the last group to 8.

A character spans at most 2 bytes; a byte spans up to 3 characters.
"""
import base64 as python_base64

from qkvc.task import Mode, Rule, Symbols, Task

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
BYTES = tuple(f'x{v:02x}' for v in range(256))


def encode_sources(n):
  """Character n starts at bit 5 * (n % 8) of its 40-bit group: in byte (5 * (n % 8)) // 8."""
  first = 5 * (n // 8) + (5 * (n % 8)) // 8
  return first, first + 1


def encode_char(a, b, place):
  """The 5 bits starting at offset (5 * place) % 8 of the 16 bits a, b."""
  return (((a << 8) | b) >> (11 - (5 * place) % 8)) & 31


def decode_sources(n):
  """Byte n starts at bit 8 * (n % 5) of its 40-bit group: in character (8 * (n % 5)) // 5."""
  first = 8 * (n // 5) + (8 * (n % 5)) // 5
  return first, first + 1, first + 2


def decode_byte(a, b, c, place):
  """The 8 bits starting at offset (8 * place) % 5 of the 15 bits a, b, c."""
  return (((a << 10) | (b << 5) | c) >> (7 - (8 * place) % 5)) & 255


def random_bytes(rng, length):
  return bytes(rng.randrange(256) for _ in range(length))


def reference(mode, data):
  if mode == 'decode':
    return [BYTES[b] for b in python_base64.b32decode(data)]
  return list(python_base64.b32encode(data).decode())


# A byte exists while its last character is not '=': byte k ends in character (8k + 7) // 5, which
# is the second or third character it reads. A character is '=' once its first byte is missing.
TASK = Task(
    name='base32',
    symbols=(Symbols('char', tuple(ALPHABET), 5), Symbols('byte', BYTES, 8)),
    flag_tokens={'pad': '='},
    modes=(
        Mode('decode', 'dec', inputs='char', outputs='byte', group=(8, 5), fetch=('first', 'second', 'third'),
             sources=decode_sources, combine=decode_byte, read=list, write=''.join,
             sample=lambda rng, length: python_base64.b32encode(random_bytes(rng, length)).decode(),
             rules=(Rule(0, 'first', 'missing', 'end', 'no more groups -> END'),
                    Rule(1, 'third', 'pad', 'end', "its last char is '=' -> END"),
                    Rule(2, 'second', 'pad', 'end', "its last char is '=' -> END"),
                    Rule(3, 'third', 'pad', 'end', "its last char is '=' -> END"),
                    Rule(4, 'second', 'pad', 'end', "its last char is '=' -> END"))),
        Mode('encode', 'enc', inputs='byte', outputs='char', group=(5, 8), fetch=('first', 'second'),
             sources=encode_sources, combine=encode_char, read=lambda data: [BYTES[b] for b in data],
             write=lambda names: bytes(int(n[1:], 16) for n in names), sample=random_bytes,
             rules=(Rule(0, 'first', 'missing', 'end', 'no more groups -> END'),
                    *(Rule(j, 'first', 'missing', 'pad', "group ran out of bytes -> '='") for j in range(2, 8))),
             pad='='),
    ),
    reference=reference,
)
