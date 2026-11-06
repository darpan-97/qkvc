# qkvc

Describe an algorithm in a short file. qkvc builds a transformer that computes it, and proves the transformer
follows that description for every valid input of any length. Nothing is trained: each weight is placed by rule
and has a name.

```
$ python -m qkvc run base64 encode "Hello Darpan!"
S G V s b G 8 g R G F y c G F u I Q = =
```

[Watch a base64 model run in your browser](https://darpan-97.github.io/qkvc/), or follow every step with real
values in the [notebooks](notebooks/).

**Status:** in active development. Support for more algorithms and more ways to customise the transformer is
coming.

- [How it works](#how-it-works)
- [Notebooks](#notebooks)
- [What is proved](#what-is-proved)
- [Supported algorithms](#supported-algorithms)
- [Limits](#limits)
- [Use](#use)
- [Add an algorithm](#add-an-algorithm)
- [Sizes and speed](#sizes-and-speed)
- [Repository](#repository)
- [References](#references)

## How it works

`Hello Darpan!` through the base64 model. Base64 turns 3 bytes into 4 characters of 6 bits:

```
H         e         l
01001000  01100101  01101100
010010  000110  010101  101100
S       G       V       s

l         o         space
01101100  01101111  00100000
011011  000110  111100  100000
b       G       8       g

D         a         r
01000100  01100001  01110010
010001  000110  000101  110010
R       G       F       y

p         a         n
01110000  01100001  01101110
011100  000110  000101  101110
c       G       F       u

!
00100001
001000  010000  (pad)   (pad)
I       Q       =       =
```

The last group has one byte, so its 8 bits are filled out with zeros to make two characters, and `=` marks the rest.

The task file gives the builder (`qkvc/build.py`) the algorithm:

```python
def encode_char(a, b, place):      # a, b: the bytes this output reads; place: its position in the group
  return [a >> 2, ((a & 3) << 4) | (b >> 4), ((a & 15) << 2) | (b >> 6), a & 63][place]
```

| In the task file | Becomes |
|---|---|
| `group=(3, 4)`: 3 bytes to 4 characters | the layout of the token stream |
| `sources`: the inputs each output reads | attention heads |
| `combine`: how an output is made | MLP units |
| `rules`: when to pad or stop | more MLP units |
| `symbols`: bytes and characters | embedding and output layer |

The result has 325 tokens, 76 numbers per token, 5 attention heads, 54 MLP units and 70,603 parameters. Writing `S`:

```
$ python -m qkvc trace base64 encode "Hello Darpan!"
output 1: S
  enc_first    reads x48 'H', 2 back
  enc_second   reads x65 'e', 1 back
  unit  encode place 0: first.bit3 -> out bit 1
  unit  encode place 0: first.bit6 -> out bit 4
```

1. Tokens are laid out in groups of 8 (`ENC H e l S G V s`), so what an output needs is always the same distance back.
2. Attention heads fetch those tokens: the query names a position, and only the token there matches.
3. MLP units act as AND gates and copy each fetched bit to its place: bits 1 and 4 make `010010`, which is 18, `S`.
4. The output layer scores each token by how many bits match, and `S` wins.
5. Rules write `=` and `END` when bytes run out, giving `SGVsbG8gRGFycGFuIQ==`.

## Notebooks

Every step, with real values. Start with [chapter 1](notebooks/01_what_we_are_building.ipynb).

| | Chapter | Shows |
|---|---|---|
| 1 | [What we are building](notebooks/01_what_we_are_building.ipynb) | base64 manually, the task file |
| 2 | [The stream](notebooks/02_the_stream.ipynb) | the groups, the distances, why any length works |
| 3 | [Named numbers](notebooks/03_named_numbers.ipynb) | the 76 numbers of a token |
| 4 | [Attention as a lookup](notebooks/04_attention_as_lookup.ipynb) | queries, keys and scores for one head |
| 5 | [Units as logic](notebooks/05_units_as_logic.ipynb) | a ReLU as an AND gate, how wires are found |
| 6 | [The output layer](notebooks/06_the_output_layer.ipynb) | every token's score, why `S` wins |
| 7 | [Padding and the end](notebooks/07_padding_and_the_end.ipynb) | how `=` and `END` win |
| 8 | [Proving it](notebooks/08_proving_it.ipynb) | the proof, and planted mistakes it catches |
| 9 | [Other algorithms](notebooks/09_other_algorithms.ipynb) | wiring, bit functions, lookup tables, state |
| 10 | [The limits](notebooks/10_the_limits.ipynb) | what the builder refuses |

`python notebooks/build.py` rebuilds them from `notebooks/src/`, running every cell.

## What is proved

| Claim | Coverage |
|---|---|
| The model does exactly what the task's rules say | every valid input of any length, read from the weights |
| The rules match the real algorithm | checked against a reference implementation, as below |

| Tasks | Rules checked on |
|---|---|
| Groups of up to 2^24 cases, such as base64 | every full group, plus sampled whole inputs |
| Larger groups | every value each output reads, plus sampled whole inputs |
| Tasks with a carry | every short input, plus sampled longer ones |

For base64 encoding, all 65,792 one- and two-byte endings are also checked ([chapter 8](notebooks/08_proving_it.ipynb)).

`python tests/controls.py` plants 11 mistakes in weights and task files. The proof rejects all 11.

## Supported algorithms

16 algorithms, all passing the proof (`python tests/tasks.py --prove`, 4 minutes on a laptop).

| Family | Algorithms | Why it fits |
|---|---|---|
| Rearranging bits | base64, base64url, base32, hex, binary, bit reversal, byte swap | each output bit copies one input bit |
| Substitution | ROT13, Caesar, Atbash, upper/lower case | each output depends on one input |
| With a key | XOR, Vigenère | the key sits at a fixed position at the start |
| Running state | binary addition, parity, XOR checksum | the previous output carries the state forward |

## Limits

The model is one attention layer and one MLP layer, with heads that look a fixed distance back. It fits
algorithms that work like a sliding window with a small carried state.

Enforced by the builder:

| Limit | Value |
|---|---|
| Distance from an output to what it reads | the same in every group |
| Input bits per output bit, for bit functions | at most 4 |
| Lookup table size | at most 4,096 units per output position |
| Layers | one attention, one MLP |

Refused when tried ([chapter 10](notebooks/10_the_limits.ipynb)):

| Algorithm | Why |
|---|---|
| Reversing a string | the distance depends on the input's length |
| Multiplying two bytes | it needs 65,536 table units |

Out of reach for the same reasons, not attempted:

| Algorithm | Why |
|---|---|
| Sorting | an output's position depends on every value |
| Search, LZ77 | where to read depends on the data |
| Brackets of any depth, counting | the state grows with the input |
| Long multiplication | each digit depends on many digits |
| SHA-256, AES | many rounds of mixing |

Also: the models are slower than ordinary code, and the Vigenère key has a fixed length of 4.

## Use

```
pip install -r requirements.txt
```

| Command | Does |
|---|---|
| `python -m qkvc run hex encode "Man"` | writes `4 d 6 1 6 e` |
| `python -m qkvc run vigenere encrypt LEMOATTACK` | writes `L X F O N O`; the first 4 letters are the key |
| `python -m qkvc run add add 1011+0110` | writes `1 0 0 0 1`, least significant digit first (17) |
| `python -m qkvc verify base64` | proves one model |
| `python -m qkvc trace base64 encode "Man"` | shows each head's read and each unit that fires |
| `python tests/tasks.py --prove` | tests and proves every task |
| `python tests/controls.py` | plants mistakes the proof must catch |

Needs numpy only.

## Add an algorithm

One file in `tasks/`. This is hex:

```python
TASK = Task(
    name='hex',
    symbols=(Symbols('digit', DIGITS, 4), Symbols('byte', BYTES, 8)),
    flag_tokens={},
    modes=(
        Mode('decode', 'dec', inputs='digit', outputs='byte', group=(2, 1), fetch=('high', 'low'),
             sources=lambda n: (2 * n, 2 * n + 1), combine=lambda high, low, place: (high << 4) | low,
             read=list, write=''.join, sample=lambda rng, length: random_bytes(rng, length).hex(),
             rules=(Rule(0, 'high', 'missing', 'end', 'no more digits -> END'),)),
        Mode('encode', 'enc', inputs='byte', outputs='digit', group=(1, 2), fetch=('byte',),
             sources=lambda n: (n // 2,), combine=lambda byte, place: (byte >> 4) if place == 0 else (byte & 15),
             read=lambda data: [BYTES[b] for b in data], write=lambda names: bytes(int(n[1:], 16) for n in names),
             sample=random_bytes,
             rules=(Rule(0, 'byte', 'missing', 'end', 'no more bytes -> END'),)),
    ),
    reference=reference,
)
```

| Field | Meaning |
|---|---|
| `sources(n)` | the input positions output `n` reads |
| `combine` | how the output is made from them |
| `rules` | when to stop or pad |
| `reference` | an ordinary implementation the proof compares against |

`qkvc/task.py` lists every field.

## Sizes and speed

| Task | Parameters | Non-zero |
|---|---:|---:|
| base64, base64url | 70,603 | 5,348 |
| base32 | 96,065 | 5,147 |
| hex | 39,168 | 4,369 |
| binary | 109,792 | 4,460 |
| byteswap | 78,718 | 5,038 |
| bits | 28,660 | 4,430 |
| rot13, atbash | 37,791 | 7,654 |
| caesar, case | 74,788 | 11,532 |
| xor | 140,917 | 5,403 |
| vigenere | 625,422 | 82,685 |
| add | 5,384 | 361 |
| parity | 760 | 63 |
| checksum | 27,160 | 4,176 |

Every task is tested on inputs of at least 20,000 symbols.

`qkvc run` computes many outputs per pass (Jacobi decoding) and gives the same result as one at a time, which
`tests/tasks.py` checks. Most tasks run 5 to 11 times faster that way; parity and the checksum do not, because
each output depends on the one before.

| 1 MiB of base64, 2 CPU threads | Time |
|---|---|
| Decode | 37 s |
| Encode | 47 s |
| One output at a time | about 5 minutes each way |

## Repository

| Path | Contents |
|---|---|
| `qkvc/` | the builder, runner, proof and trace |
| `tasks/` | the 16 algorithms |
| `tests/` | task tests and planted-mistake controls |
| `notebooks/` | the 10 chapters; `src/` holds their sources |
| `web/` | the builder's base64 model in a browser, [live here](https://darpan-97.github.io/qkvc/); locally: `python -m http.server 8797 --directory web`, then http://127.0.0.1:8797/ |

`web/export.py` writes the page's weights from the builder, and `node web/check_engine.js` holds the page equal to
the Python model.

In research this is called compiling ([Tracr](https://arxiv.org/abs/2301.05062) compiles RASP programs into
transformer weights). Here the input is a Python description with no language to parse, so it is called building.

## References

| Used for | Source |
|---|---|
| Base64 itself | [RFC 4648](https://datatracker.ietf.org/doc/html/rfc4648) |
| Transformers | Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017 |
| Attention as a lookup | Weiss et al., [Thinking Like Transformers](https://arxiv.org/abs/2106.06981), ICML 2021 |
| Weights from a description; lookup-table units | Lindner et al., [Tracr](https://arxiv.org/abs/2301.05062), NeurIPS 2023 |
| The residual stream as named numbers | Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/), 2021 |
| Units as logic gates | McCulloch and Pitts, [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://doi.org/10.1007/BF02478259), 1943; Nair and Hinton, [ReLU](https://mlanthology.org/icml/2010/nair2010icml-rectified), ICML 2010 |
| Attention over a sliding window | Beltagy et al., [Longformer](https://arxiv.org/abs/2004.05150), 2020; Jiang et al., [Mistral 7B](https://arxiv.org/abs/2310.06825), 2023 |
| Proving from the weights' structure | Gross et al., [Compact Proofs of Model Performance via Mechanistic Interpretability](https://arxiv.org/abs/2406.11779), NeurIPS 2024 |
| Planting mistakes to test a check | DeMillo et al., [Hints on Test Data Selection](https://doi.org/10.1109/C-M.1978.218136), 1978 |
| Many outputs per pass | Song et al., [Accelerating Feedforward Computation via Parallel Nonlinear Equation Solving](https://arxiv.org/abs/2002.03629), ICML 2021; Santilli et al., [Accelerating Transformer Inference for Translation via Parallel Decoding](https://arxiv.org/abs/2305.10427), ACL 2023 |
| The page's layout | Cho et al., [Transformer Explainer](https://arxiv.org/abs/2408.04619), 2024 |

## Licence

MIT ([`LICENSE`](LICENSE)).
