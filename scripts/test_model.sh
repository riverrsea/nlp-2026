#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

CONFIG_PATH="${1:-}"
CHECKPOINT_PATH="${2:-}"

if [[ -z "$CONFIG_PATH" || -z "$CHECKPOINT_PATH" ]]; then
  echo "Usage: bash scripts/test_model.sh <config_path> <checkpoint_path> [extra args]" >&2
  exit 1
fi

shift 2

"$PYTHON_BIN" src/main.py \
  --config "$CONFIG_PATH" \
  --mode test \
  --data_dir data \
  --output_dir outputs \
  --device auto \
  --checkpoint "$CHECKPOINT_PATH" \
  "$@"
