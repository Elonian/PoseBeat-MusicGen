#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_motion_adapter import main as train_main


def _has_config_arg(argv: list[str]) -> bool:
    return any(arg == "--config" or arg.startswith("--config=") for arg in argv)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not _has_config_arg(args):
        args = ["--config", "configs/motion_to_music_hires.yaml", *args]
    train_main(args)


if __name__ == "__main__":
    main()
