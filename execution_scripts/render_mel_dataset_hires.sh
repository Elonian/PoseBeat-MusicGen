#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/motion_to_music_hires.yaml}"
NUM_PROC="${NUM_PROC:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/mntdatalora/src/PoseBeat-MusicGen/data/motion_to_music_aistpp_legacy_2026-05-26/data_and_model/input music/aistpp_hires_sorted}"

echo "[PoseBeat] Rendering mel dataset: config=${CONFIG}"
python scripts/render_mel_dataset.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-proc "${NUM_PROC}" \
  "$@"
