#!/usr/bin/env python3
"""Inspect the frozen GUI using PyInstaller's archive reader (audit tool only)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
from PyInstaller.archive.readers import CArchiveReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('gui', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.gui.resolve()
exe = root / 'loong_gui_sim'
a = CArchiveReader(str(exe))
outer = sorted(a.toc)
modules = []
for name in outer:
    if a.toc[name][-1] == 'z':
        modules.extend(a.open_embedded_archive(name).toc)
files = []
links = []
for p in sorted((root / '_internal').rglob('*')):
    if p.is_symlink():
        if not p.exists() or root not in p.resolve().parents:
            raise SystemExit(f'Invalid link: {p}')
        links.append({'path': str(p.relative_to(root)), 'target': str(p.readlink())})
    elif p.is_file():
        files.append({'path': str(p.relative_to(root)), 'size': p.stat().st_size,
                      'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
base = root / '_internal/base_library.zip'
with zipfile.ZipFile(base) as z:
    stdlib = sorted(z.namelist())
forbidden = ('PyQt5', 'PyQt6', 'PySide2', 'paramiko', 'readline', 'cryptography',
             'nacl', 'cffi', '_cffi_backend', 'pycparser', 'bcrypt')
identifiers = outer + sorted(modules) + stdlib + [f['path'] for f in files]
hits = {n: [v for v in identifiers if re.search(r'(^|[/\.])' + re.escape(n) + r'([/\.]|$)', v)]
        for n in forbidden}
if any(hits.values()):
    raise SystemExit(f'Unexpected legacy components: {hits}')
if any(n == 'PySide6' or n.startswith('PySide6.') or n == 'shiboken6'
       or n.startswith('shiboken6.') for n in modules):
    raise SystemExit('LGPL binding Python entry points must remain externally replaceable')
for name in ('PySide6/__init__.py', 'shiboken6/__init__.py'):
    if not (root / '_internal' / name).is_file():
        raise SystemExit(f'Missing replaceable binding entry: {name}')
result = {'schema': 1, 'format': 'PyInstaller onedir',
          'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
          'executable_size': exe.stat().st_size, 'outer_entries': outer,
          'pyz_modules': sorted(modules), 'base_library_entries': stdlib,
          'external_files': files, 'symlinks': links, 'legacy_module_hits': hits,
          'scope': 'GUI package inventory; not a license opinion, vulnerability scan or full SDK SBOM'}
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(f'OK: {len(files)} external files, {len(links)} symlinks; legacy bindings absent; LGPL entries external')
