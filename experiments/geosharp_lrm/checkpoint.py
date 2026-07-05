"""Checkpoint save/resume utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when wrapped by DDP."""
    return model.module if hasattr(model, "module") else model


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    """Find the latest step checkpoint in a run directory."""
    path = Path(out_dir)
    if not path.exists():
        return None
    candidates = []
    for item in path.glob("step_*.pt"):
        match = re.match(r"step_(\d+)\.pt", item.name)
        if match:
            candidates.append((int(match.group(1)), item))
    if not candidates:
        last = path / "last.pt"
        return last if last.exists() else None
    return max(candidates, key=lambda item: item[0])[1]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: dict[str, Any],
) -> None:
    """Atomically save training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> int:
    """Load training state and return the stored step."""
    ckpt = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if "rng" in ckpt:
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    return int(ckpt.get("step", 0))


def prune_checkpoints(out_dir: str | Path, keep: int) -> None:
    """Keep only the newest N step checkpoints."""
    if keep <= 0:
        return
    path = Path(out_dir)
    checkpoints = []
    for item in path.glob("step_*.pt"):
        match = re.match(r"step_(\d+)\.pt", item.name)
        if match:
            checkpoints.append((int(match.group(1)), item))
    for _, item in sorted(checkpoints, reverse=True)[keep:]:
        item.unlink(missing_ok=True)


