"""Small production-friendly metric logging utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch


class JsonlLogger:
    """Append metrics as JSON lines."""

    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, object]) -> None:
        if not self.enabled:
            return
        payload = {"time": time.time(), **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


class MetricAverager:
    """Accumulate scalar metrics over a window."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, torch.Tensor | float]) -> None:
        self.count += 1
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = float(value.detach().float().cpu())
            self.totals[key] = self.totals.get(key, 0.0) + float(value)

    def pop(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        values = {key: value / self.count for key, value in self.totals.items()}
        self.totals.clear()
        self.count = 0
        return values


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    """Compute PSNR from an RGB MSE tensor."""
    return -10.0 * torch.log10(mse.clamp_min(1.0e-8))

