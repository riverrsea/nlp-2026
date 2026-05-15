#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

"$PYTHON_BIN" src/main.py \
  --config cfg/prompt_gpt.yaml \
  --mode train \
  --data_dir data \
  --output_dir outputs \
  --device auto \
  "$@"
