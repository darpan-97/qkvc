"""Carry: each output also reads the model's own previous output, so a value runs along the whole
chain (a carry, a running checksum). A fetch named in from_outputs reads an earlier output; the
proof's induction over output positions then does real work: output n is right because output
n - 1 was."""
