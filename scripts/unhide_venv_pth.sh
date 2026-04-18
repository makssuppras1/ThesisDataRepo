#!/usr/bin/env bash
# Python 3.11+ skips .pth files with the macOS UF_HIDDEN flag; editable installs then break.
# Run from repo root after sync/install if you see: ModuleNotFoundError: No module named 'thesisdatarepo'
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
if [[ ! -d "$VENV" ]]; then
  echo "No .venv at $VENV" >&2
  exit 1
fi
for pydir in "$VENV"/lib/python*/site-packages; do
  [[ -d "$pydir" ]] || continue
  find "$pydir" -maxdepth 1 -name '*.pth' -exec chflags nohidden {} \; 2>/dev/null || true
done
echo "Ran chflags nohidden on all *.pth under $VENV/lib/python*/site-packages."
