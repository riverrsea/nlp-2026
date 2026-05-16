#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-pretrained_models/bert-base-chinese}"
FEW_SHOT_K="${FEW_SHOT_K:-0}"

"$PYTHON_BIN" src/main.py \
  --config cfg/prompt_bert.yaml \
  --mode train \
  --data_dir data \
  --output_dir outputs \
  --device auto \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --few_shot_k "$FEW_SHOT_K" \
  "$@"
