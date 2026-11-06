"""qkvc from the command line.

    python -m qkvc run TASK MODE INPUT         write the answer, e.g.  python -m qkvc run hex encode "Man"
    python -m qkvc verify TASK                 the proof, link by link
    python -m qkvc trace TASK MODE INPUT       what the heads read and which units fire, output by output
"""
import runpy
import sys

COMMANDS = {'run': 'qkvc.run', 'verify': 'qkvc.verify', 'trace': 'qkvc.trace'}

if __name__ == '__main__':
  command, sys.argv = (sys.argv[1] if len(sys.argv) > 1 else ''), sys.argv[:1] + sys.argv[2:]
  if command not in COMMANDS:
    sys.exit(__doc__)
  runpy.run_module(COMMANDS[command], run_name='__main__')
