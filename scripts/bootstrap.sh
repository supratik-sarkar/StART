#!/usr/bin/env bash
# Bootstrap a development environment for StART.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3.12}
VENV=${VENV_DIR:-.venv}

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -e ".[dev]"
start doctor
echo "Bootstrap complete. Activate with: source $VENV/bin/activate"
