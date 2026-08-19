#!/usr/bin/env bash
# Convenience wrapper. All installation logic lives in scripts/bootstrap.py so
# there is exactly one implementation — a shell script with its own dependency
# logic is how macOS and Linux instructions drift apart.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" scripts/bootstrap.py "$@"
