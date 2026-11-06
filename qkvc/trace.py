"""What the model does at each output: what its heads read, and which MLP units fired.

    python -m qkvc trace base64 encode "Hello Darpan!"
"""
import sys

from qkvc.run import load


def shown(name):
  """a byte token as it reads: x48 is 'H'"""
  if len(name) == 3 and name[0] == 'x':
    v = int(name[1:], 16)
    return f"{name} {chr(v)!r}" if 32 <= v < 127 else name
  return name


def main():
  if len(sys.argv) != 4:
    sys.exit(__doc__)
  task, mode, text = sys.argv[1:]
  model = load(task)
  m = model.r.task.mode(mode)
  data = text.encode() if m.inputs == 'byte' else text
  written = []
  for n, step in enumerate(model.steps(mode, data, detail=True)):
    written.append(step['name'])
    print(f"output {n + 1}: {step['name']}")
    for head, (token, back) in step['reads'].items():
      if head.startswith(m.flag + '_'):
        print(f'  {head:<12} reads {shown(token)}, {back} back')
    for unit in step['units']:
      print(f'  unit  {unit}')
  print('written:', ''.join(w for w in written if w != 'END'))


if __name__ == '__main__':
  main()
