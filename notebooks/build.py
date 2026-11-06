"""Build the notebooks: each src/NN_name.py is run, cell by cell, and written as NN_name.ipynb with every
cell's output saved, so the notebooks can be read without running them and every output shown comes from a
real run. A cell that fails stops the build: a notebook with an error in it is never written.

    python notebooks/build.py [NN ...]        (all of them, or the ones whose number is given)

A source file is Python with cells marked as in Jupytext's "percent" format:

    # %% [markdown]
    # # A heading
    # Text, each line after "# ".
    # %%
    code()

A code cell's last line, if it is an expression, is shown as Jupyter shows it.
"""
import ast
import contextlib
import io
import json
import os
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent


def cells(text):
  out, kind, lines = [], None, []
  for line in text.splitlines():
    if line.startswith('# %%'):
      if kind:
        out.append((kind, lines))
      kind, lines = ('markdown' if '[markdown]' in line else 'code'), []
    elif kind:
      lines.append(line)
  if kind:
    out.append((kind, lines))
  result = []
  for kind, lines in out:
    while lines and not lines[-1].strip():
      lines.pop()
    while lines and not lines[0].strip():
      lines.pop(0)
    if kind == 'markdown':
      lines = [l[2:] if l.startswith('# ') else l[1:] if l.startswith('#') else l for l in lines]
    if lines:
      result.append((kind, '\n'.join(lines)))
  return result


def run(code, env):
  """Runs a cell; returns its output text and the shown value of its last expression (or None)."""
  tree = ast.parse(code)
  last = None
  if tree.body and isinstance(tree.body[-1], ast.Expr):
    last = ast.Expression(tree.body.pop().value)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    exec(compile(tree, '<cell>', 'exec'), env)
    value = eval(compile(last, '<cell>', 'eval'), env) if last else None
  return buf.getvalue(), None if value is None else repr(value)


def source(text):
  lines = text.split('\n')
  return [l + '\n' for l in lines[:-1]] + [lines[-1]]


def build(path):
  env = {'__name__': '__main__'}
  nb_cells, n = [], 0
  os.chdir(HERE)
  start = time.time()
  for kind, text in cells(path.read_text(encoding='utf-8')):
    if kind == 'markdown':
      nb_cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source(text)})
      continue
    n += 1
    try:
      out, value = run(text, env)
    except Exception:
      print(f'{path.name}: cell {n} failed:\n{text}\n', file=sys.stderr)
      traceback.print_exc()
      sys.exit(1)
    outputs = []
    if out:
      outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': source(out.rstrip('\n') + '\n')})
    if value is not None:
      outputs.append({'output_type': 'execute_result', 'execution_count': n, 'metadata': {},
                      'data': {'text/plain': source(value)}})
    nb_cells.append({'cell_type': 'code', 'execution_count': n, 'metadata': {}, 'outputs': outputs,
                     'source': source(text)})
  kernel = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
  nb = {'cells': nb_cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': kernel, 'language_info': {'name': 'python'}}}
  target = path.parent.parent / (path.stem + '.ipynb')
  target.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
  print(f'{target.name}: {n} code cells run, {time.time() - start:.1f} s')


def main():
  wanted = sys.argv[1:]
  for path in sorted((HERE / 'src').glob('*.py')):
    if not wanted or path.name.split('_')[0] in wanted:
      build(path)


if __name__ == '__main__':
  main()
