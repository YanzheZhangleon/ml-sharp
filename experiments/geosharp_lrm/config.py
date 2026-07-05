"""Validated configuration objects for GeoSHARP-LRM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.geosharp_lrm.losses import LossConfig
from experiments.geosharp_lrm.model import GeoSHARPLRMConfig


@dataclass
class DataConfig:
    """Dataset and sampling configuration."""

    manifest_path: str = "data/geosharp_manifest.json"
    image_root: str = "data"
    val_manifest_path: str | None = None
    val_image_root: str | None = None
    num_input_views: int = 4
    num_target_views: int = 1
    shuffle: bool = True
    strict: bool = True


@dataclass
class TrainingConfig:
    """Training runtime configuration."""

    device: str = "cuda"
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 4
    max_steps: int = 100_000
    lr: float = 2.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    amp: bool = True
    amp_dtype: str = "bf16"
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    log_every: int = 10
    save_every: int = 1_000
    val_every: int = 2_000
    val_batches: int = 16
    keep_last_checkpoints: int = 5
    resume: str | None = None
    out_dir: str = "experiments/geosharp_lrm/runs/base"
    compile_model: bool = False
    find_unused_parameters: bool = False


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    model: GeoSHARPLRMConfig = field(default_factory=GeoSHARPLRMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        training = dict(raw.get("training", {}))
        if "betas" in training:
            training["betas"] = tuple(training["betas"])
        config = cls(
            model=GeoSHARPLRMConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            loss=LossConfig(**raw.get("loss", {})),
            training=TrainingConfig(**training),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Fail early for configuration mistakes that would otherwise waste runs."""
        if self.model.image_height % self.model.patch_size != 0:
            raise ValueError("model.image_height must be divisible by model.patch_size.")
        if self.model.image_width % self.model.patch_size != 0:
            raise ValueError("model.image_width must be divisible by model.patch_size.")
        if self.data.num_input_views < 1:
            raise ValueError("data.num_input_views must be positive.")
        if self.data.num_target_views < 1:
            raise ValueError("data.num_target_views must be positive for supervised rendering.")
        if self.data.num_input_views > self.model.max_input_views:
            raise ValueError("data.num_input_views exceeds model.max_input_views.")
        if self.training.grad_accum_steps < 1:
            raise ValueError("training.grad_accum_steps must be positive.")
        if self.training.amp_dtype not in {"bf16", "fp16"}:
            raise ValueError("training.amp_dtype must be 'bf16' or 'fp16'.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation."""
        return _to_plain_dict(self)


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    return value

