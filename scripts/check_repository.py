#!/usr/bin/env python3
"""Check release inventory in a full checkout or an LFS-pointer checkout.

This does not execute the GUI or training, fetch LFS, or certify licenses.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
POINTER = re.compile(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n')


def check(root):
    root = root.resolve()
    manifest = json.loads((root / 'FILE_HASHES.json').read_text())
    errors = []
    pointers = 0
    for name, expected in manifest.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or
                name == 'FILE_HASHES.json' or not re.fullmatch('[0-9a-f]{64}', expected)):
            errors.append('Invalid inventory entry: ' + name)
            continue
        path = root / relative
        if root not in path.resolve().parents:
            errors.append('External file/link: ' + name)
            continue
        if not path.is_file():
            errors.append('Missing: ' + name)
            continue
        with path.open('rb') as stream:
            first = stream.read(256)
            if first.startswith(b'version https://git-lfs.github.com/spec/v1'):
                match = POINTER.fullmatch(first)
                if match is None or match[1].decode() != expected or int(match[2]) <= 0:
                    errors.append('Invalid LFS pointer/object hash: ' + name)
                else:
                    pointers += 1
            else:
                digest = hashlib.sha256(first)
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                if digest.hexdigest() != expected:
                    errors.append('Modified: ' + name)
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True)
    if tracked.returncode == 0:
        files = set(tracked.stdout.decode().rstrip('\0').split('\0'))
        expected_files = set(manifest) | {'FILE_HASHES.json'}
        for name in sorted(files - expected_files):
            errors.append('Tracked file absent from inventory: ' + name)
        for name in sorted(expected_files - files):
            errors.append('Inventory file absent from Git: ' + name)
    for name in ('LICENSE', 'LICENSES.md', 'NOTICE', 'CONTRIBUTING.md',
                 'docs/reporting-bugs.md', 'docs/repository-maintenance.md',
                 '.github/ISSUE_TEMPLATE/bug_report.yml',
                 '.gitcode/ISSUE_TEMPLATE/bug_report.yml',
                 '.atomgit/ISSUE_TEMPLATE/bug_report.md',
                 'training_walking/uv.lock', 'training_dance/requirements.txt',
                 'training_parkour/requirements-training-cu128.lock'):
        if name not in manifest:
            errors.append('Required document/dependency inventory missing: ' + name)
    for name in ('start_sim.sh', 'train.sh', 'training_dance/install.sh',
                 'training_dance/train.sh', 'training_parkour/scripts/install_training.sh',
                 'training_parkour/scripts/train_lite.sh'):
        result = subprocess.run(['bash', '-n', str(root / name)], capture_output=True)
        if result.returncode:
            errors.append('Shell syntax check failed: ' + name)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print('OK: {} inventory entries; {} LFS pointers verified; documents and shell syntax passed'.format(len(manifest), pointers))
    if pointers:
        print('LFS objects were not downloaded. Run git lfs pull and check_release.py before packaging.')
    return 0


if __name__ == '__main__':
    sys.exit(check(ROOT))
