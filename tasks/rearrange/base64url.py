"""base64url (RFC 4648 section 5): base64 with '-' and '_' in place of '+' and '/', for URLs and
file names. The same groups, rules and padding as base64."""
import base64 as python_base64
import dataclasses

from qkvc.task import Symbols
from tasks.rearrange import base64

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'


def reference(mode, data):
  if mode == 'decode':
    return [base64.BYTES[b] for b in python_base64.urlsafe_b64decode(data)]
  return list(python_base64.urlsafe_b64encode(data).decode())


decode, encode = base64.TASK.modes
TASK = dataclasses.replace(
    base64.TASK,
    name='base64url',
    symbols=(Symbols('char', tuple(ALPHABET), 6), Symbols('byte', base64.BYTES, 8)),
    modes=(dataclasses.replace(decode, sample=lambda rng, length: python_base64.urlsafe_b64encode(
        base64.random_bytes(rng, length)).decode()), encode),
    reference=reference,
)
