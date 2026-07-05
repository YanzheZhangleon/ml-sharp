"""Generic multi-view manifest dataset for GeoSHARP-LRM."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def _load_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = pil_to_tensor(image).float() / 255.0
    tensor = F.interpolate(tensor[None], size=size, mode="bilinear", align_corners=False)[0]
    return tensor


def _scale_intrinsics(fxfycxcy: torch.Tensor, original_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
    oh, ow = original_hw
    th, tw = target_hw
    out = fxfycxcy.clone()
    out[0] *= tw / ow
    out[2] *= tw / ow
    out[1] *= th / oh
    out[3] *= th / oh
    return out


class MultiViewSceneDataset(Dataset):
    """Loads scenes from a compact JSON manifest.

    Manifest format:
    [
      {
        "scene_id": "scene_a",
        "frames": [
          {
            "image": "scene_a/images/000.png",
            "c2w": [[...], [...], [...], [...]],
            "intrinsics": [fx, fy, cx, cy],
            "height": 1080,
            "width": 1920
          }
        ]
      }
    ]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        image_size: tuple[int, int],
        num_input_views: int,
        num_target_views: int,
        shuffle: bool = True,
        strict: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.shuffle = shuffle
        self.strict = strict
        self.scenes: list[dict[str, Any]] = json.loads(self.manifest_path.read_text())
        self._validate_manifest()
        self.scenes = [
            scene
            for scene in self.scenes
            if len(scene["frames"]) >= self.num_input_views + self.num_target_views
        ]
        if not self.scenes:
            raise ValueError("No scene has enough frames for the requested input/target split.")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene = self.scenes[index]
        frames = list(scene["frames"])
        if self.shuffle:
            selected = random.sample(frames, self.num_input_views + self.num_target_views)
        else:
            selected = frames[: self.num_input_views + self.num_target_views]
        input_frames = selected[: self.num_input_views]
        target_frames = selected[self.num_input_views :]
        return {
            "input": self._load_frames(input_frames),
            "target": self._load_frames(target_frames),
            "scene_id": scene.get("scene_id", str(index)),
        }

    def _load_frames(self, frames: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images, intrinsics, c2ws = [], [], []
        for frame in frames:
            image_path = self.image_root / frame["image"]
            original_hw = (int(frame["height"]), int(frame["width"]))
            images.append(_load_image(image_path, self.image_size))
            k = torch.tensor(frame["intrinsics"], dtype=torch.float32)
            intrinsics.append(_scale_intrinsics(k, original_hw, self.image_size))
            c2ws.append(torch.tensor(frame["c2w"], dtype=torch.float32))
        return {
            "image": torch.stack(images),
            "fxfycxcy": torch.stack(intrinsics),
            "c2w": torch.stack(c2ws),
        }

    def _validate_manifest(self) -> None:
        if not isinstance(self.scenes, list):
            raise ValueError(f"{self.manifest_path} must contain a list of scenes.")
        for scene_idx, scene in enumerate(self.scenes):
            frames = scene.get("frames")
            if not isinstance(frames, list):
                raise ValueError(f"Scene {scene_idx} is missing a frames list.")
            for frame_idx, frame in enumerate(frames):
                missing = {"image", "height", "width", "intrinsics", "c2w"} - set(frame)
                if missing:
                    raise ValueError(f"Scene {scene_idx} frame {frame_idx} missing keys: {sorted(missing)}")
                image_path = self.image_root / frame["image"]
                if self.strict and not image_path.exists():
                    raise FileNotFoundError(f"Missing image: {image_path}")
                if len(frame["intrinsics"]) != 4:
                    raise ValueError(f"Scene {scene_idx} frame {frame_idx} intrinsics must be [fx, fy, cx, cy].")
                if len(frame["c2w"]) != 4 or any(len(row) != 4 for row in frame["c2w"]):
                    raise ValueError(f"Scene {scene_idx} frame {frame_idx} c2w must be 4x4.")
