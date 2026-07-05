"""Rendering and regularization losses for GeoSHARP-LRM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer


@dataclass
class LossConfig:
    """Loss weights and renderer settings."""

    rgb_l1_weight: float = 1.0
    rgb_l2_weight: float = 0.2
    ssim_weight: float = 0.0
    opacity_weight: float = 1.0e-4
    scale_weight: float = 1.0e-4
    scale_anisotropy_weight: float = 1.0e-5
    background: str = "white"
    low_pass_filter_eps: float = 1.0e-2
    target_chunk_size: int = 1


def invert_c2w(c2w: torch.Tensor) -> torch.Tensor:
    """Convert c2w to w2c view matrices."""
    return torch.linalg.inv(c2w)


def intrinsics_matrix(fxfycxcy: torch.Tensor) -> torch.Tensor:
    """Convert fx/fy/cx/cy to 3x3 intrinsics."""
    k = torch.zeros(*fxfycxcy.shape[:-1], 3, 3, device=fxfycxcy.device, dtype=fxfycxcy.dtype)
    k[..., 0, 0] = fxfycxcy[..., 0]
    k[..., 1, 1] = fxfycxcy[..., 1]
    k[..., 0, 2] = fxfycxcy[..., 2]
    k[..., 1, 2] = fxfycxcy[..., 3]
    k[..., 2, 2] = 1.0
    return k


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    """Compute PSNR from image MSE."""
    return -10.0 * torch.log10(mse.clamp_min(1.0e-8))


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Small differentiable SSIM loss for RGB tensors in [0, 1]."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred.flatten(0, 1), 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target.flatten(0, 1), 3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred.flatten(0, 1) ** 2, 3, stride=1, padding=1) - mu_x**2
    sigma_y = F.avg_pool2d(target.flatten(0, 1) ** 2, 3, stride=1, padding=1) - mu_y**2
    sigma_xy = F.avg_pool2d(pred.flatten(0, 1) * target.flatten(0, 1), 3, stride=1, padding=1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ((1.0 - ssim.clamp(0.0, 1.0)) * 0.5).mean()


class GeoRenderLoss(nn.Module):
    """Render predicted Gaussians into target cameras and compute image losses."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.renderer = GSplatRenderer(
            color_space="sRGB",
            background_color=config.background,  # type: ignore[arg-type]
            low_pass_filter_eps=config.low_pass_filter_eps,
        )

    def render_targets(
        self,
        gaussians: Gaussians3D,
        target_fxfycxcy: torch.Tensor,
        target_c2w: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        """Render all target views, chunking along the target-view axis."""
        _, target_views, _ = target_fxfycxcy.shape
        rendered = []
        chunk = max(1, self.config.target_chunk_size)
        for start in range(0, target_views, chunk):
            end = min(start + chunk, target_views)
            chunk_colors = []
            for target_idx in range(start, end):
                viewmats = invert_c2w(target_c2w[:, target_idx])
                ks = intrinsics_matrix(target_fxfycxcy[:, target_idx])
                out = self.renderer(gaussians, viewmats, ks, image_width=image_width, image_height=image_height)
                chunk_colors.append(out.color)
            rendered.extend(chunk_colors)
        return torch.stack(rendered, dim=1)

    def forward(
        self,
        gaussians: Gaussians3D,
        target_images: torch.Tensor,
        target_fxfycxcy: torch.Tensor,
        target_c2w: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, _, _, h, w = target_images.shape
        pred = self.render_targets(gaussians, target_fxfycxcy, target_c2w, h, w)
        l1 = F.l1_loss(pred, target_images)
        l2 = F.mse_loss(pred, target_images)
        ssim = ssim_loss(pred, target_images) if self.config.ssim_weight > 0 else l2.new_zeros(())
        opacity_reg = gaussians.opacities.mean()
        scale_reg = gaussians.singular_values.mean()
        scale_aniso = (gaussians.singular_values.max(dim=-1).values / gaussians.singular_values.min(dim=-1).values.clamp_min(1.0e-6)).mean()
        loss = (
            self.config.rgb_l1_weight * l1
            + self.config.rgb_l2_weight * l2
            + self.config.ssim_weight * ssim
            + self.config.opacity_weight * opacity_reg
            + self.config.scale_weight * scale_reg
            + self.config.scale_anisotropy_weight * scale_aniso
        )
        metrics = {
            "loss": loss.detach(),
            "rgb_l1": l1.detach(),
            "rgb_l2": l2.detach(),
            "ssim_loss": ssim.detach(),
            "psnr": psnr_from_mse(l2.detach()),
            "opacity_reg": opacity_reg.detach(),
            "scale_reg": scale_reg.detach(),
            "scale_anisotropy": scale_aniso.detach(),
            "render_mean": pred.mean().detach(),
        }
        return loss, metrics
