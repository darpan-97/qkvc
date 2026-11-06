"""Task files, one per algorithm, in a folder per family. A family is the kind of computation the
builder needs for it:

    rearrange/    every output bit is a copy of one input bit (base64, hex, base32, ...)
    substitute/   each output is any function of one input symbol (ROT13, Caesar, Atbash, case)
    key/          each output combines a message symbol with a key symbol (XOR, Vigenere)
    carry/        each output also reads the previous output (addition, parity, XOR checksum)
"""
import importlib
import pkgutil


def every_task():
  """(family, task) for every task file, family by family."""
  found = []
  for info in pkgutil.walk_packages(__path__, __name__ + '.'):
    if not info.ispkg:
      found.append((info.name.split('.')[1], importlib.import_module(info.name).TASK))
  return found


def find(name):
  """The task called `name`, in any family."""
  for _, task in every_task():
    if task.name == name:
      return task
  raise KeyError(f'no task called {name!r}: ' + ', '.join(t.name for _, t in every_task()))
