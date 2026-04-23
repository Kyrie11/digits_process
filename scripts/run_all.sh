#!/usr/bin/env bash
set -euo pipefail

MAT_PATH=${1:-data/digits4000.mat}
OUT_DIR=${2:-outputs/course_digits_full}
CONFIG=${3:-configs/base.yaml}

python main.py inspect --config "$CONFIG" --mat_path "$MAT_PATH" --output_dir "$OUT_DIR"
python main.py run --config "$CONFIG" --mat_path "$MAT_PATH" --output_dir "$OUT_DIR"
python main.py report --config "$CONFIG" --mat_path "$MAT_PATH" --output_dir "$OUT_DIR"
