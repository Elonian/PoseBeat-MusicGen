#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/motion_to_music_hires.yaml}"
NUM_PROC="${NUM_PROC:-4}"
RECONSTRUCTION_N_ITER="${RECONSTRUCTION_N_ITER:-4}"

echo "[PoseBeat] Building full hires mel dataset: config=${CONFIG}"
python scripts/build_full_hires_mel_dataset.py \
  --config "${CONFIG}" \
  --num-proc "${NUM_PROC}" \
  --reconstruction-n-iter "${RECONSTRUCTION_N_ITER}" \
  "$@"
