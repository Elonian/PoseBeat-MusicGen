#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/motion_to_music_hires.yaml}"
GPUS="${GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-3}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export NCCL_P2P_DISABLE
export NCCL_IB_DISABLE

echo "[PoseBeat] Installing Python requirements from requirements.txt"
python -m pip install -r requirements.txt

effective_batch_size=$((GPUS * BATCH_SIZE))
echo "[PoseBeat] Training config: config=${CONFIG}"
echo "[PoseBeat] Launch: gpus=${GPUS} batch_size_per_gpu=${BATCH_SIZE} effective_batch_size=${effective_batch_size} num_workers_per_gpu=${NUM_WORKERS}"
echo "[PoseBeat] NCCL: NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "[PoseBeat] Logs: logs/motion_to_music_hires/train_motion_adapter.log plus rank logs when using torchrun"

train_args=(
  --config "${CONFIG}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)

if [[ "${GPUS}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node "${GPUS}" scripts/train_motion_adapter_hires.py "${train_args[@]}" "$@"
else
  python scripts/train_motion_adapter_hires.py "${train_args[@]}" "$@"
fi
