"""Production training entrypoint for GeoSHARP-LRM without render-error refinement."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from experiments.geosharp_lrm.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from experiments.geosharp_lrm.config import ExperimentConfig
from experiments.geosharp_lrm.dataset import MultiViewSceneDataset
from experiments.geosharp_lrm.distributed import (
    DistributedState,
    barrier,
    cleanup,
    init_distributed,
    reduce_metrics,
    wrap_ddp,
)
from experiments.geosharp_lrm.logging_utils import JsonlLogger, MetricAverager
from experiments.geosharp_lrm.losses import GeoRenderLoss
from experiments.geosharp_lrm.model import GeoSHARPLRM


def seed_everything(seed: int, rank: int) -> None:
    """Seed Python, NumPy, and PyTorch per rank."""
    seed = seed + rank * 10_003
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(obj: Any, device: torch.device) -> Any:
    """Move nested tensors to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_device(value, device) for value in obj]
    return obj


def build_dataset(config: ExperimentConfig, split: str) -> MultiViewSceneDataset | None:
    """Build train or validation dataset."""
    data = config.data
    if split == "train":
        manifest_path = data.manifest_path
        image_root = data.image_root
        shuffle = data.shuffle
    else:
        if data.val_manifest_path is None:
            return None
        manifest_path = data.val_manifest_path
        image_root = data.val_image_root or data.image_root
        shuffle = False
    return MultiViewSceneDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        image_size=(config.model.image_height, config.model.image_width),
        num_input_views=data.num_input_views,
        num_target_views=data.num_target_views,
        shuffle=shuffle,
        strict=data.strict,
    )


def build_loader(
    dataset: MultiViewSceneDataset,
    config: ExperimentConfig,
    state: DistributedState,
    split: str,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Create dataloader and optional distributed sampler."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=state.world_size,
        rank=state.rank,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
    ) if state.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(sampler is None and split == "train"),
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=(state.device.type == "cuda"),
        persistent_workers=config.training.num_workers > 0,
        drop_last=(split == "train"),
    )
    return loader, sampler


def forward_loss(
    model: torch.nn.Module,
    loss_fn: GeoRenderLoss,
    batch: dict[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run model and rendering loss for one batch."""
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        outputs = model(
            batch["input"]["image"],
            batch["input"]["fxfycxcy"],
            batch["input"]["c2w"],
        )
        loss, metrics = loss_fn(
            outputs.gaussians,
            batch["target"]["image"],
            batch["target"]["fxfycxcy"],
            batch["target"]["c2w"],
        )
    metrics = {**metrics, **outputs.diagnostics}
    return loss, metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loss_fn: GeoRenderLoss,
    loader: DataLoader | None,
    config: ExperimentConfig,
    state: DistributedState,
) -> dict[str, float]:
    """Run a bounded validation pass."""
    if loader is None:
        return {}
    model.eval()
    amp_enabled = config.training.amp and state.device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.training.amp_dtype == "bf16" else torch.float16
    avg = MetricAverager()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= config.training.val_batches:
            break
        batch = to_device(batch, state.device)
        _, metrics = forward_loss(model, loss_fn, batch, amp_enabled, amp_dtype, state.device)
        metrics = reduce_metrics(metrics, state)
        avg.update(metrics)
    model.train()
    return avg.pop()


def train(config_path: Path) -> None:
    """Train GeoSHARP-LRM."""
    config = ExperimentConfig.from_yaml(config_path)
    state = init_distributed(config.training.device)
    seed_everything(config.training.seed, state.rank)

    out_dir = Path(config.training.out_dir)
    if state.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.resolved.json").write_text(json.dumps(config.to_dict(), indent=2))
    barrier()

    train_dataset = build_dataset(config, "train")
    assert train_dataset is not None
    val_dataset = build_dataset(config, "val")
    train_loader, train_sampler = build_loader(train_dataset, config, state, "train")
    val_loader = build_loader(val_dataset, config, state, "val")[0] if val_dataset is not None else None

    model = GeoSHARPLRM(config.model).to(state.device)
    if config.training.compile_model:
        model = torch.compile(model)
    model = wrap_ddp(model, state, config.training.find_unused_parameters)
    loss_fn = GeoRenderLoss(config.loss).to(state.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler(state.device.type, enabled=config.training.amp and state.device.type == "cuda")

    start_step = 0
    resume_path = config.training.resume
    if resume_path == "auto":
        latest = latest_checkpoint(out_dir)
        resume_path = str(latest) if latest is not None else None
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer, scaler, map_location=state.device)

    train_logger = JsonlLogger(out_dir / "train_metrics.jsonl", enabled=state.is_main)
    val_logger = JsonlLogger(out_dir / "val_metrics.jsonl", enabled=state.is_main)
    avg = MetricAverager()
    step = start_step
    epoch = 0
    amp_enabled = config.training.amp and state.device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.training.amp_dtype == "bf16" else torch.float16
    progress = tqdm(total=config.training.max_steps, initial=step, disable=not state.is_main, desc="GeoSHARP-LRM")

    try:
        model.train()
        while step < config.training.max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch += 1
            for batch_idx, batch in enumerate(train_loader):
                batch = to_device(batch, state.device)
                should_sync = ((batch_idx + 1) % config.training.grad_accum_steps) == 0
                ddp_context = model.no_sync() if state.distributed and not should_sync else nullcontext()
                with ddp_context:
                    loss, metrics = forward_loss(model, loss_fn, batch, amp_enabled, amp_dtype, state.device)
                    scaled_loss = loss / config.training.grad_accum_steps
                    scaler.scale(scaled_loss).backward()
                if not should_sync:
                    continue

                if config.training.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                metrics = reduce_metrics(metrics, state)
                avg.update(metrics)

                if state.is_main and step % config.training.log_every == 0:
                    window = avg.pop()
                    window["step"] = step
                    window["lr"] = optimizer.param_groups[0]["lr"]
                    train_logger.log(window)
                    progress.set_postfix({key: round(value, 5) for key, value in window.items() if isinstance(value, float)})

                if step % config.training.val_every == 0:
                    val_metrics = validate(model, loss_fn, val_loader, config, state)
                    if state.is_main and val_metrics:
                        val_metrics["step"] = step
                        val_logger.log(val_metrics)

                if state.is_main and step % config.training.save_every == 0:
                    save_checkpoint(out_dir / f"step_{step:07d}.pt", model, optimizer, scaler, step, config.to_dict())
                    save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, step, config.to_dict())
                    prune_checkpoints(out_dir, config.training.keep_last_checkpoints)

                progress.update(1)
                if step >= config.training.max_steps:
                    break
        if state.is_main:
            save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, step, config.to_dict())
    finally:
        progress.close()
        cleanup()


class nullcontext:
    """Tiny contextlib.nullcontext equivalent to keep imports minimal."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/geosharp_lrm/config.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

