"""Build a disposable collector-capable TCP tree from a verified project revision."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def matches(root, expected):
    for rel, digest in expected.items():
        path = root / rel
        if digest is None:
            if path.exists():
                return False
        elif not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    source, target = args.project.resolve(), args.target.resolve()
    kit = Path(__file__).resolve().parent
    spec = json.loads((kit / 'sources.json').read_text())
    revision = next((name for name, files in spec.items() if matches(source, files)), None)
    if revision is None:
        raise SystemExit('Input revision differs from both published bases. Compare sources.json and port the patch manually.')
    if target.exists() or target.is_relative_to(source):
        raise SystemExit('Choose a fresh output directory outside the input project.')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        staging = Path(tmp) / 'tree'
        for sub in ('adaptive_planning', 'contracts', 'configs'):
            shutil.copytree(source/sub, staging/sub, ignore=shutil.ignore_patterns('__pycache__','node_modules','.DS_Store'))
        if revision != 'v3':
            with (kit / f'from_{revision}.patch').open('rb') as patch:
                subprocess.run(['patch','-s','-p1'], cwd=staging, stdin=patch, check=True)
        if not matches(staging, spec['v3']):
            raise RuntimeError('Patched module digest mismatch')
        staging.rename(target)
    print(f'{revision} -> v3; verified {target}')

if __name__ == '__main__':
    main()
