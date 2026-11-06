"""base64 (RFC 4648): every 3 bytes as 4 characters of 6 bits, '=' padding the last group."""
import base64 as python_base64

from qkvc.task import Mode, Rule, Symbols, Task

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
BYTES = tuple(f'x{v:02x}' for v in range(256))


def decode_sources(n):
  """Output byte n reads characters 4 * (n // 3) + n % 3 and the one after it."""
  first = 4 * (n // 3) + n % 3
  return first, first + 1


def decode_byte(a, b, place):
  return [(a << 2) | (b >> 4), ((a & 15) << 4) | (b >> 2), ((a & 3) << 6) | b][place] & 255


def encode_sources(n):
  """Output character n reads bytes 3 * (n // 4) + (0, 0, 1, 2)[n % 4] and the one after it."""
  first = 3 * (n // 4) + (0, 0, 1, 2)[n % 4]
  return first, first + 1


def encode_char(a, b, place):
  return [a >> 2, ((a & 3) << 4) | (b >> 4), ((a & 15) << 2) | (b >> 6), a & 63][place]


def random_bytes(rng, length):
  return bytes(rng.randrange(256) for _ in range(length))


def reference(mode, data):
  if mode == 'decode':
    return [BYTES[b] for b in python_base64.b64decode(data)]
  return list(python_base64.b64encode(data).decode())


TASK = Task(
    name='base64',
    symbols=(Symbols('char', tuple(ALPHABET), 6), Symbols('byte', BYTES, 8)),
    flag_tokens={'pad': '='},
    modes=(
        Mode('decode', 'dec', inputs='char', outputs='byte', group=(4, 3), fetch=('first', 'second'),
             sources=decode_sources, combine=decode_byte, read=list, write=''.join,
             sample=lambda rng, length: python_base64.b64encode(random_bytes(rng, length)).decode(),
             rules=(Rule(0, 'first', 'missing', 'end', 'no more groups -> END'),
                    Rule(1, 'second', 'pad', 'end', "second char is '=' -> END"),
                    Rule(2, 'second', 'pad', 'end', "second char is '=' -> END"))),
        Mode('encode', 'enc', inputs='byte', outputs='char', group=(3, 4), fetch=('first', 'second'),
             sources=encode_sources, combine=encode_char, read=lambda data: [BYTES[b] for b in data],
             write=lambda names: bytes(int(n[1:], 16) for n in names), sample=random_bytes,
             rules=(Rule(0, 'first', 'missing', 'end', 'no more groups -> END'),
                    Rule(2, 'first', 'missing', 'pad', "group ran out of bytes -> '='"),
                    Rule(3, 'first', 'missing', 'pad', "group ran out of bytes -> '='")),
             pad='='),
    ),
    reference=reference,
)
