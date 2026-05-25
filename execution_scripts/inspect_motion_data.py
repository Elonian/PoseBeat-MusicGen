#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.motion_dataset import infer_motion_dim, load_motion_encodings


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a DMD motion encoding pickle.")
    parser.add_argument("--motion-pickle", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    encodings = load_motion_encodings(args.motion_pickle)
    print(f"items: {len(encodings)}")
    print(f"motion_dim: {infer_motion_dim(encodings)}")
    for key in list(encodings)[: args.limit]:
        value = np.asarray(encodings[key])
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
