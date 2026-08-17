# -*- coding: utf-8 -*-
"""Verify that the device-facing cells of EchoPath_Jetson.ipynb could run on the board.

You cannot make a Windows environment BE the Jetson: JetPack 4.6 is Python 3.6 with
numpy 1.19.5 and OpenCV 4.1.1, and none of those have Python 3.11 wheels. What you can
do is check that the code you write here stays inside what the board can execute — and
find out now rather than after flashing.

Two checks, over the cells that actually run on the device:

  1. SYNTAX   Python 3.6 only. No dataclasses (3.7), walrus (3.8), f"{x=}" (3.8),
              builtin generics like list[int] (3.9), or match statements (3.10).

  2. IMPORTS  Nothing outside the device's package set at MODULE level. Imports inside
              a function or behind `if BACKEND == 'torch':` are fine — they never
              execute on the board. That distinction is what lets one notebook serve
              both machines.

Usage:
    python check_jetson_compat.py                 # exit 0 = clean, 1 = problems
    python check_jetson_compat.py --verbose
"""
from __future__ import print_function

import ast
import io
import json
import os
import re
import sys

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EchoPath_Jetson.ipynb')

# Cell indices that execute on the Jetson. Cells 10/12/14/16/20/22 are workstation-only
# (matplotlib, video files, ffmpeg) and are deliberately not checked.
DEVICE_CELLS = [2, 4, 6, 8, 18, 24]

# Available on the board: JetPack supplies tensorrt/cv2/numpy, pip supplies the rest.
DEVICE_PACKAGES = {
    'numpy', 'cv2', 'tensorrt', 'pycuda', 'pyttsx3',
}

STDLIB = {
    'os', 'sys', 'platform', 'subprocess', 'threading', 'time', 'queue', 'json',
    'math', 're', 'collections', 'tempfile', 'shutil', 'wave', 'pathlib', 'argparse',
    'itertools', 'functools', 'traceback', 'warnings', 'glob', 'random', 'struct',
    'ctypes', 'signal', 'logging', 'datetime', 'copy', 'io', 'errno', 'socket',
}

ALLOWED = DEVICE_PACKAGES | STDLIB

# Regex checks for syntax the AST cannot flag portably from a 3.11 parser.
SYNTAX_PATTERNS = [
    (r'@dataclass|from\s+dataclasses', 'dataclasses (3.7+)'),
    (r'f["\'][^"\']*\{[^}]*=\}', 'f-string = specifier (3.8+)'),
    (r'(?<![:=!<>]):=(?!=)', 'walrus operator (3.8+)'),
    (r'->\s*(list|dict|tuple|set|type)\[', 'builtin generic in return type (3.9+)'),
    (r':\s*(list|dict|tuple|set)\[[A-Za-z]', 'builtin generic annotation (3.9+)'),
    (r'^\s*match\s+\w+\s*:\s*$', 'match statement (3.10+)'),
]


def module_level_imports(tree):
    """Imports that execute unconditionally at import time.

    An import nested inside a function, class, try, or an `if BACKEND == ...` branch
    is not module level - it only runs if that path is taken, which on the board it
    is not. Those are how the torch backend stays out of the device's way.
    """
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name.split('.')[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append((node.module.split('.')[0], node.lineno))
    return out


def check_cell(idx, src, verbose=False):
    problems = []

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [('cell %d' % idx, 'does not parse: %s' % e)]

    for name, lineno in module_level_imports(tree):
        if name not in ALLOWED:
            problems.append(
                ('cell %d line %d' % (idx, lineno),
                 "module-level 'import %s' - not available on the Jetson. Move it "
                 "inside a function or behind `if BACKEND == 'torch':`" % name))
        elif verbose:
            print('    ok  import %s' % name)

    for pattern, why in SYNTAX_PATTERNS:
        for m in re.finditer(pattern, src, re.M):
            line = src[:m.start()].count('\n') + 1
            snippet = m.group(0).strip()[:40]
            problems.append(('cell %d line %d' % (idx, line),
                             '%s -> %r' % (why, snippet)))
    return problems


def main():
    verbose = '--verbose' in sys.argv

    if not os.path.exists(NB):
        print('notebook not found: %s' % NB)
        return 2
    nb = json.load(io.open(NB, encoding='utf-8'))
    cells = nb['cells']

    print('EchoPath - Jetson compatibility check')
    print('notebook : %s' % os.path.basename(NB))
    print('checking : cells %s (device path)'
          % ', '.join(str(c) for c in DEVICE_CELLS))
    print('target   : Python 3.6 / JetPack 4.6 / aarch64')
    print('-' * 72)

    all_problems = []
    for idx in DEVICE_CELLS:
        if idx >= len(cells) or cells[idx]['cell_type'] != 'code':
            all_problems.append(('cell %d' % idx, 'missing or not a code cell - '
                                 'DEVICE_CELLS is out of date'))
            continue
        src = ''.join(cells[idx]['source'])
        if verbose:
            print('  cell %d (%d lines)' % (idx, src.count('\n')))
        problems = check_cell(idx, src, verbose)
        all_problems.extend(problems)
        print('  cell %-3d %s' % (idx, 'OK' if not problems else
                                  '%d problem(s)' % len(problems)))

    print('-' * 72)
    if all_problems:
        print('%d problem(s):\n' % len(all_problems))
        for where, what in all_problems:
            print('  %-22s %s' % (where, what))
        print('\nFAIL - this would not run on the Jetson.')
        return 1

    print('PASS - the device path is Python 3.6 clean and imports nothing the board '
          'does not have.')
    print('\nNote: this checks the CODE, not the runtime. Version differences that are')
    print('not syntax - numpy 1.19 vs 1.26, OpenCV 4.1 vs 4.11 - can still bite, and')
    print('only running on the board will find those.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
