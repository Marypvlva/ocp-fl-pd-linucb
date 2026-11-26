#!/usr/bin/env bash
# Run from the repository. This changes only the selected VM.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
if [[ ${1:-} != --no-system ]]; then
  [[ $(id -u) == 0 ]] || { echo 'Run sudo bash deploy/timeweb/bootstrap.sh'; exit 1; }
  source /etc/os-release
  [[ $ID == ubuntu && $VERSION_ID == 24.04 && $(uname -m) == x86_64 ]] || { echo 'Requires Ubuntu 24.04 x86_64'; exit 1; }
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y python3-venv ca-certificates curl git openssl chrony rsync jq tmux
  systemctl enable --now chrony
fi
[[ -f requirements-linux-x86_64.lock ]] || { echo 'Missing verified Linux lock file'; exit 1; }
mkdir -p .bootstrap
if [[ ! -x .bootstrap/bin/uv ]]; then
  python3 -m venv .bootstrap
  .bootstrap/bin/python -m pip install --disable-pip-version-check uv==0.9.11
fi
UV="$ROOT/.bootstrap/bin/uv"
"$UV" python install 3.13.7
if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,13)' >/dev/null 2>&1; then
  mv .venv ".venv-before-timeweb-$(date +%Y%m%d%H%M%S)"
fi
if [[ ! -x .venv/bin/python ]]; then "$UV" venv --python 3.13.7 .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,13), "Choose a new .venv directory for Python 3.13"'
"$UV" pip sync --python .venv/bin/python --require-hashes requirements-linux-x86_64.lock
"$UV" pip install --python .venv/bin/python --no-deps -e .
"$UV" pip check --python .venv/bin/python
mkdir -p results/bootstrap
.venv/bin/python -m pip --version > /dev/null 2>&1 || true
.venv/bin/python -c 'import platform,numpy,scipy,web3; print(platform.platform(),platform.python_version())' | tee results/bootstrap/platform.txt
sha256sum requirements-linux-x86_64.lock | tee results/bootstrap/lock.sha256
printf 'Installed. Use %s/.venv/bin/python\n' "$ROOT"
