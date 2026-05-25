from __future__ import annotations

import json
import logging as py_logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


def setup_logging(
    log_dir: str | Path,
    *,
    name: str = "posebeat",
    filename: str = "run.log",
    level: str = "INFO",
    reset_handlers: bool = True,
) -> py_logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    logger = py_logging.getLogger(name)
    logger.setLevel(getattr(py_logging, level.upper()))
    logger.propagate = False

    if reset_handlers:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    formatter = py_logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = py_logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = py_logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("logging to %s", log_path)
    return logger


def log_config(logger: py_logging.Logger, config: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(config, sort_keys=True, default_flow_style=False).strip()
    logger.info("config:\n%s", rendered)


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


class JsonlMetricLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, **metrics: Any) -> None:
        payload = {"time": time.time()}
        payload.update(metrics)
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class RunningAverage:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += count

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
