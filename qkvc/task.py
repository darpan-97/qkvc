"""The task format: an algorithm described for qkvc, with no weights in it.

A task is a set of modes (directions, such as decode and encode). In each mode, outputs come in
groups, and every output is computed from a few symbols at positions that depend only on the
output's index:

    sources(n)                 the positions output n reads, one per fetch label
    combine(*values, place)    the output's value, from those symbols' values and its place in the group

A fetch normally reads an input position; a fetch named in `from_outputs` reads an earlier output
instead (sources gives its output index, such as n - 1), which is how a carry runs along a chain. A
position before the start or past the end reads as missing, with value 0.

Rules say when an output is END or the pad symbol instead, from what was fetched. A mode may start its
input with a header that every group reads, such as a key; the groups follow it.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Symbols:
  """Tokens that carry a value: names[i] has value i, held in `bits` bits. `kind` names the flag
  is_<kind> that marks them in the stream."""
  kind: str
  names: tuple
  bits: int


@dataclass(frozen=True)
class Rule:
  """At `place`, if fetched symbol `head` has `flag` ('missing' or a flag token's kind), and every
  (head, flag) in `also` holds too, the output is `writes` ('end' or 'pad')."""
  place: int
  head: str
  flag: str
  writes: str
  text: str
  also: tuple = ()

  @property
  def conditions(self):
    return ((self.head, self.flag),) + tuple(self.also)


@dataclass(frozen=True)
class Mode:
  name: str                 # e.g. 'decode'
  flag: str                 # e.g. 'dec': token DEC, flags is_dec and mode_dec, heads dec_<label>
  inputs: str               # the Symbols kind read
  outputs: str              # the Symbols kind written
  group: tuple              # (inputs per group, outputs per group)
  fetch: tuple              # a label for each fetched symbol, e.g. ('first', 'second')
  sources: Callable         # n -> positions, one per fetch label
  combine: Callable         # (*values, place) -> output value; must also work on numpy arrays of values
  read: Callable            # the mode's input -> token names
  write: Callable           # token names -> the mode's input (the inverse of read)
  sample: Callable          # (random.Random, length) -> a valid input built from `length` units
  rules: tuple = ()
  pad: str = None           # the token a 'pad' rule writes
  header: int = 0           # leading input symbols every group reads, such as a key
  from_outputs: tuple = ()  # fetch labels whose sources are earlier outputs, not inputs
  extra: int = 0            # outputs after the last group, such as a final carry

  def reads_output(self, i):
    return self.fetch[i] in self.from_outputs


@dataclass(frozen=True)
class Task:
  name: str
  symbols: tuple            # Symbols, in token order
  flag_tokens: dict         # kind -> token name, for tokens with no value, e.g. {'pad': '='}
  modes: tuple
  reference: Callable       # (mode name, input) -> the output's token names, END not included

  def mode(self, name):
    return next(m for m in self.modes if m.name == name)

  @property
  def value_bits(self):
    return max(s.bits for s in self.symbols)
