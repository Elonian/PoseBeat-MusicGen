from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_training_checkpoint(
    path: str | Path,
    *,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int,
    epoch: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "adapter": adapter.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_adapter_state(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
