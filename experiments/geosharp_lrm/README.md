# GeoSHARP-LRM

Production-oriented training stack for the no-refinement model requested here:

```text
iLRM/MVP-style multi-view transformer
+ ReSplat-style monocular depth and multi-view matching priors
+ SHARP-style high-resolution residual Gaussian decoder
```

This version intentionally excludes render-error recurrent refinement. The training code is built so that the geometry frontend can be replaced by ReSplat/UniMatch without changing the dataset, loss, checkpoint, or distributed training pipeline.

## What Is Implemented

- Validated YAML configuration with early error checks.
- Strict multi-view manifest dataset with camera/intrinsics validation.
- Geometry frontend:
  - monocular depth branch
  - inverse-depth plane-sweep multi-view matching
  - confidence map and geometry feature output
- iLRM-style transformer backbone with per-view local attention and global multi-view fusion.
- SHARP-style dense Gaussian residual decoder:
  - depth/ray based Gaussian placement
  - residual xyz/scale/quaternion/color/opacity heads
  - zero-initialized residual decoder head
- gsplat-based supervised target-view rendering loss.
- L1/L2/SSIM image losses, opacity/scale/anisotropy regularization, PSNR metric.
- AMP, DDP via `torchrun`, gradient accumulation, gradient clipping.
- Resume, latest checkpoint discovery, checkpoint pruning, JSONL metric logs.
- Train/validation split support.

## Manifest Format

Create train/val manifests, for example `data/geosharp_manifest_train.json`:

```json
[
  {
    "scene_id": "scene_a",
    "frames": [
      {
        "image": "scene_a/images/000000.png",
        "height": 1080,
        "width": 1920,
        "intrinsics": [1200.0, 1200.0, 960.0, 540.0],
        "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      }
    ]
  }
]
```

`intrinsics` are `[fx, fy, cx, cy]` in the original image resolution. The loader rescales them to the training resolution.

## Single-GPU Training

```powershell
.venv\Scripts\python.exe -m experiments.geosharp_lrm.train --config experiments\geosharp_lrm\config.yaml
```

## Multi-GPU Training

```powershell
torchrun --nproc_per_node=4 -m experiments.geosharp_lrm.train --config experiments\geosharp_lrm\config.yaml
```

## Resume

Set this in config:

```yaml
training:
  resume: auto
```

or point to a specific checkpoint:

```yaml
training:
  resume: experiments/geosharp_lrm/runs/base/step_0005000.pt
```

## Logs and Checkpoints

Each run directory contains:

- `config.resolved.json`
- `train_metrics.jsonl`
- `val_metrics.jsonl`
- `last.pt`
- `step_*.pt`

## Replacing the Geometry Frontend With ReSplat UniMatch

Keep this interface stable:

```python
GeometryOutputs(depth, confidence, feature)
```

Replace `DepthMatchingFrontend` in `model.py` with a wrapper around ReSplat's `MultiViewUniMatch`, then map its outputs to:

- `depth`: metric per-view depth, shape `[B, V, 1, H, W]`
- `confidence`: matching confidence, shape `[B, V, 1, H, W]`
- `feature`: geometry feature map, shape `[B, V, C, H, W]`

The rest of the stack does not need to change.
