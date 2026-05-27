#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python execution_scripts/sample_motion_adapter.py --config configs/motion_audio_adapter.yaml --all-cdcd "$@"
