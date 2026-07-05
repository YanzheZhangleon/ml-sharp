"""GeoSHARP-LRM: iLRM-style transformer with depth/matching guided Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from sharp.utils.gaussians import Gaussians3D


class GeometryOutputs(NamedTuple):
    """Geometry priors used by the transformer and Gaussian decoder."""

    depth: torch.Tensor
    confidence: torch.Tensor
    feature: torch.Tensor


class ModelOutputs(NamedTuple):
    """Forward outputs used by training."""

    gaussians: Gaussians3D
    geometry: GeometryOutputs
    diagnostics: dict[str, torch.Tensor]


@dataclass
class GeoSHARPLRMConfig:
    """Model hyperparameters."""

    image_height: int = 544
    image_width: int = 960
    max_input_views: int = 32
    depth_min: float = 0.2
    depth_max: float = 500.0
    match_height: int = 68
    match_width: int = 120
    depth_bins: int = 48
    inverse_depth_bins: bool = True
    geometry_backend: str = "plane_sweep"
    match_temperature: float = 0.05
    geo_feature_dim: int = 24
    patch_size: int = 8
    transformer_dim: int = 512
    transformer_heads: int = 8
    local_layers: int = 2
    global_layers: int = 8
    decoder_dim: int = 256
    gaussian_layers: int = 1
    xyz_residual_scale: float = 0.02
    scale_multiplier: float = 0.7
    opacity_bias: float = -2.0
    min_scale: float = 1.0e-5
    max_scale: float = 10.0


def _intrinsics_to_matrix(fxfycxcy: torch.Tensor) -> torch.Tensor:
    """Convert fx/fy/cx/cy to 3x3 camera intrinsics."""
    k = torch.zeros(*fxfycxcy.shape[:-1], 3, 3, device=fxfycxcy.device, dtype=fxfycxcy.dtype)
    k[..., 0, 0] = fxfycxcy[..., 0]
    k[..., 1, 1] = fxfycxcy[..., 1]
    k[..., 0, 2] = fxfycxcy[..., 2]
    k[..., 1, 2] = fxfycxcy[..., 3]
    k[..., 2, 2] = 1.0
    return k


def scale_intrinsics(
    fxfycxcy: torch.Tensor, source_hw: tuple[int, int], target_hw: tuple[int, int]
) -> torch.Tensor:
    """Scale fx/fy/cx/cy from source image size to target image size."""
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    out = fxfycxcy.clone()
    out[..., 0] *= target_w / source_w
    out[..., 2] *= target_w / source_w
    out[..., 1] *= target_h / source_h
    out[..., 3] *= target_h / source_h
    return out


def compute_rays(
    fxfycxcy: torch.Tensor, c2w: torch.Tensor, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute world-space ray origins and normalized directions."""
    b, v, _ = fxfycxcy.shape
    device = fxfycxcy.device
    dtype = fxfycxcy.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = fxfycxcy[..., 0].reshape(b * v, 1, 1)
    fy = fxfycxcy[..., 1].reshape(b * v, 1, 1)
    cx = fxfycxcy[..., 2].reshape(b * v, 1, 1)
    cy = fxfycxcy[..., 3].reshape(b * v, 1, 1)
    x = ((x + 0.5)[None] - cx) / fx
    y = ((y + 0.5)[None] - cy) / fy
    z = torch.ones_like(x)
    dirs_cam = torch.stack([x, y, z], dim=1).flatten(2)
    c2w_flat = c2w.reshape(b * v, 4, 4)
    dirs_world = torch.bmm(c2w_flat[:, :3, :3], dirs_cam)
    dirs_world = F.normalize(dirs_world, dim=1).reshape(b, v, 3, height, width)
    origins = c2w_flat[:, :3, 3].reshape(b, v, 3, 1, 1).expand_as(dirs_world)
    return origins, dirs_world


class ConvBlock(nn.Module):
    """Small convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthMatchingFrontend(nn.Module):
    """Lightweight ReSplat-style geometry frontend.

    It predicts a per-view monocular depth and refines it with a small plane-sweep
    multi-view matching volume. The interface mirrors heavier matchers such as
    MultiViewUniMatch so it can be swapped later.
    """

    def __init__(self, config: GeoSHARPLRMConfig) -> None:
        super().__init__()
        self.config = config
        if config.geometry_backend != "plane_sweep":
            raise ValueError(f"Unsupported geometry_backend: {config.geometry_backend}")
        c = config.geo_feature_dim
        self.feature_net = nn.Sequential(
            ConvBlock(3, c),
            ConvBlock(c, c),
            ConvBlock(c, c),
        )
        self.depth_head = nn.Sequential(
            ConvBlock(c, c),
            nn.Conv2d(c, 1, 1),
        )

    def forward(
        self, images: torch.Tensor, fxfycxcy: torch.Tensor, c2w: torch.Tensor
    ) -> GeometryOutputs:
        b, v, _, h, w = images.shape
        mh, mw = self.config.match_height, self.config.match_width
        images_low = F.interpolate(
            images.flatten(0, 1), size=(mh, mw), mode="bilinear", align_corners=False
        )
        feats = self.feature_net(images_low).unflatten(0, (b, v))
        mono_depth_low = self._predict_monodepth(feats)
        match_depth_low, confidence_low = self._plane_sweep(feats, fxfycxcy, c2w, (h, w))
        depth_low = confidence_low * match_depth_low + (1.0 - confidence_low) * mono_depth_low
        depth = F.interpolate(
            depth_low.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False
        ).unflatten(0, (b, v))
        confidence = F.interpolate(
            confidence_low.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False
        ).unflatten(0, (b, v))
        feature = F.interpolate(
            feats.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False
        ).unflatten(0, (b, v))
        return GeometryOutputs(depth=depth, confidence=confidence, feature=feature)

    def _predict_monodepth(self, feats: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = feats.shape
        raw = self.depth_head(feats.flatten(0, 1)).unflatten(0, (b, v))
        depth_range = self.config.depth_max - self.config.depth_min
        return self.config.depth_min + torch.sigmoid(raw) * depth_range

    def _plane_sweep(
        self,
        feats: torch.Tensor,
        fxfycxcy_full: torch.Tensor,
        c2w: torch.Tensor,
        full_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, v, _, h, w = feats.shape
        if v == 1:
            depth = self._predict_monodepth(feats)
            confidence = torch.zeros(b, v, 1, h, w, device=feats.device, dtype=feats.dtype)
            return depth, confidence

        fxfycxcy = scale_intrinsics(fxfycxcy_full, full_hw, (h, w))
        if self.config.inverse_depth_bins:
            inv_bins = torch.linspace(
                1.0 / self.config.depth_max,
                1.0 / self.config.depth_min,
                self.config.depth_bins,
                device=feats.device,
                dtype=feats.dtype,
            )
            depth_bins = 1.0 / inv_bins.flip(0).clamp_min(1.0e-8)
        else:
            depth_bins = torch.linspace(
                self.config.depth_min,
                self.config.depth_max,
                self.config.depth_bins,
                device=feats.device,
                dtype=feats.dtype,
            )
        costs_per_view: list[torch.Tensor] = []

        for ref_idx in range(v):
            ref_feat = feats[:, ref_idx]
            ref_origin, ref_ray = compute_rays(
                fxfycxcy[:, ref_idx : ref_idx + 1],
                c2w[:, ref_idx : ref_idx + 1],
                h,
                w,
            )
            ref_origin = ref_origin[:, 0]
            ref_ray = ref_ray[:, 0]
            ref_costs: list[torch.Tensor] = []
            for depth in depth_bins:
                points_world = ref_origin + ref_ray * depth
                cost_sum = torch.zeros(b, 1, h, w, device=feats.device, dtype=feats.dtype)
                valid_sum = torch.zeros_like(cost_sum)
                for src_idx in range(v):
                    if src_idx == ref_idx:
                        continue
                    warped, valid = self._warp_source(feats[:, src_idx], points_world, fxfycxcy[:, src_idx], c2w[:, src_idx])
                    cost = (ref_feat - warped).square().mean(dim=1, keepdim=True)
                    cost_sum = cost_sum + cost * valid
                    valid_sum = valid_sum + valid
                ref_costs.append(cost_sum / valid_sum.clamp_min(1.0e-4))
            costs_per_view.append(torch.stack(ref_costs, dim=2))

        costs = torch.stack(costs_per_view, dim=1).squeeze(2)
        probs = torch.softmax(-costs / self.config.match_temperature, dim=2)
        bins = depth_bins.view(1, 1, -1, 1, 1)
        depth = (probs * bins).sum(dim=2, keepdim=False).unsqueeze(2)
        confidence = probs.max(dim=2, keepdim=False).values.unsqueeze(2)
        return depth, confidence

    @staticmethod
    def _warp_source(
        src_feat: torch.Tensor, points_world: torch.Tensor, src_fxfycxcy: torch.Tensor, src_c2w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = src_feat.shape
        w2c = torch.linalg.inv(src_c2w)
        points = points_world.flatten(2)
        points_h = torch.cat([points, torch.ones(b, 1, h * w, device=points.device, dtype=points.dtype)], dim=1)
        cam = torch.bmm(w2c, points_h)[:, :3]
        z = cam[:, 2].clamp_min(1.0e-4)
        u = src_fxfycxcy[:, 0, None] * (cam[:, 0] / z) + src_fxfycxcy[:, 2, None]
        v = src_fxfycxcy[:, 1, None] * (cam[:, 1] / z) + src_fxfycxcy[:, 3, None]
        x_norm = 2.0 * (u / max(w - 1, 1)) - 1.0
        y_norm = 2.0 * (v / max(h - 1, 1)) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).reshape(b, h, w, 2)
        valid = ((grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0) & (cam[:, 2].reshape(b, h, w) > 0)).to(src_feat.dtype)
        warped = F.grid_sample(src_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return warped, valid[:, None]


class PatchTransformerBackbone(nn.Module):
    """iLRM-style token backbone with local per-view and global multi-view attention."""

    def __init__(self, config: GeoSHARPLRMConfig, in_channels: int) -> None:
        super().__init__()
        self.config = config
        self.patch = nn.Conv2d(in_channels, config.transformer_dim, config.patch_size, stride=config.patch_size)
        self.view_embed = nn.Parameter(torch.zeros(1, config.max_input_views, 1, config.transformer_dim))
        local_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        global_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local = nn.TransformerEncoder(local_layer, num_layers=config.local_layers)
        self.global_fuse = nn.TransformerEncoder(global_layer, num_layers=config.global_layers)

    def forward(self, token_image: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = token_image.shape
        x = self.patch(token_image.flatten(0, 1))
        hp, wp = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2).unflatten(0, (b, v))
        x = x + self.view_embed[:, :v]
        x = self.local(x.flatten(0, 1)).unflatten(0, (b, v))
        x = self.global_fuse(x.flatten(1, 2)).unflatten(1, (v, hp * wp))
        return x.flatten(0, 1).transpose(1, 2).reshape(b, v, self.config.transformer_dim, hp, wp)


class HighResGaussianDecoder(nn.Module):
    """SHARP-style high-resolution residual Gaussian decoder."""

    def __init__(self, config: GeoSHARPLRMConfig, conditioning_channels: int) -> None:
        super().__init__()
        self.config = config
        in_channels = config.transformer_dim + conditioning_channels
        self.net = nn.Sequential(
            ConvBlock(in_channels, config.decoder_dim),
            ConvBlock(config.decoder_dim, config.decoder_dim),
            ConvBlock(config.decoder_dim, config.decoder_dim),
            nn.Conv2d(config.decoder_dim, config.gaussian_layers * 14, 1),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        token_map: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        confidence: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        fxfycxcy: torch.Tensor,
    ) -> Gaussians3D:
        b, v, _, h, w = images.shape
        token_up = F.interpolate(
            token_map.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False
        ).unflatten(0, (b, v))
        conditioning = torch.cat([images, depth, confidence, ray_d], dim=2)
        x = torch.cat([token_up, conditioning], dim=2)
        attrs = self.net(x.flatten(0, 1)).unflatten(0, (b, v))
        attrs = attrs.reshape(b, v, self.config.gaussian_layers, 14, h, w)
        xyz_delta, scale_delta, quat_delta, color_delta, opacity_delta = torch.split(
            attrs, [3, 3, 4, 3, 1], dim=3
        )

        ray_o = ray_o[:, :, None]
        ray_d = ray_d[:, :, None]
        depth_l = depth[:, :, None]
        base_xyz = ray_o + depth_l * ray_d
        xyz = base_xyz + torch.tanh(xyz_delta) * depth_l * self.config.xyz_residual_scale

        pixel_scale = self._pixel_scale(depth, fxfycxcy)[:, :, None]
        base_scale = torch.cat([pixel_scale, pixel_scale, pixel_scale * 0.25], dim=3)
        scale = base_scale * torch.exp(scale_delta.clamp(-6.0, 2.0)) * self.config.scale_multiplier
        scale = scale.clamp(self.config.min_scale, self.config.max_scale)

        quat_base = torch.zeros_like(quat_delta)
        quat_base[:, :, :, 0] = 1.0
        quat = F.normalize(quat_base + 0.1 * quat_delta, dim=3)
        color = (images[:, :, None] + 0.1 * torch.tanh(color_delta)).clamp(0.0, 1.0)
        opacity = torch.sigmoid(opacity_delta + self.config.opacity_bias)

        def flatten_attr(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 1, 2, 4, 5, 3).reshape(b, v * self.config.gaussian_layers * h * w, -1)

        return Gaussians3D(
            mean_vectors=flatten_attr(xyz),
            singular_values=flatten_attr(scale),
            quaternions=flatten_attr(quat),
            colors=flatten_attr(color),
            opacities=flatten_attr(opacity).squeeze(-1),
        )

    @staticmethod
    def _pixel_scale(depth: torch.Tensor, fxfycxcy: torch.Tensor) -> torch.Tensor:
        focal = torch.sqrt(fxfycxcy[..., 0] * fxfycxcy[..., 1]).view(*depth.shape[:2], 1, 1, 1)
        return depth / focal.clamp_min(1.0)


class GeoSHARPLRM(nn.Module):
    """Trainable no-refinement model requested by the user."""

    def __init__(self, config: GeoSHARPLRMConfig) -> None:
        super().__init__()
        self.config = config
        self.geometry = DepthMatchingFrontend(config)
        token_channels = 3 + 3 + 3 + 3 + 1 + 1 + config.geo_feature_dim
        self.backbone = PatchTransformerBackbone(config, token_channels)
        self.decoder = HighResGaussianDecoder(config, conditioning_channels=3 + 1 + 1 + 3)

    def forward(self, images: torch.Tensor, fxfycxcy: torch.Tensor, c2w: torch.Tensor) -> ModelOutputs:
        b, v, _, h, w = images.shape
        geometry = self.geometry(images, fxfycxcy, c2w)
        ray_o, ray_d = compute_rays(fxfycxcy, c2w, h, w)
        token_image = torch.cat(
            [
                images,
                ray_o,
                ray_d,
                torch.cross(ray_o, ray_d, dim=2),
                geometry.depth,
                geometry.confidence,
                geometry.feature,
            ],
            dim=2,
        )
        token_map = self.backbone(token_image)
        gaussians = self.decoder(
            token_map,
            images,
            geometry.depth,
            geometry.confidence,
            ray_o,
            ray_d,
            fxfycxcy,
        )
        diagnostics = {
            "depth_mean": geometry.depth.mean().detach(),
            "match_confidence": geometry.confidence.mean().detach(),
            "num_gaussians": torch.tensor(gaussians.mean_vectors.shape[1], device=images.device),
        }
        return ModelOutputs(gaussians=gaussians, geometry=geometry, diagnostics=diagnostics)




