#!/usr/bin/env bash
set -euo pipefail

python tools/make_synthetic_digits_mat.py --out data/synthetic_digits4000.mat
python main.py inspect --config configs/tiny_debug.yaml --mat_path data/synthetic_digits4000.mat --output_dir outputs/smoke_test
python main.py run --config configs/tiny_debug.yaml --mat_path data/synthetic_digits4000.mat --output_dir outputs/smoke_test
python main.py report --config configs/tiny_debug.yaml --mat_path data/synthetic_digits4000.mat --output_dir outputs/smoke_test
