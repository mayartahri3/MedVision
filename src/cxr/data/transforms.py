"""MONAI preprocessing + augmentation pipelines.

Input to every pipeline: float32 grayscale HxW array in [0, 1] (see io.read_xray).
Output: torch tensor C x S x S, normalised for the chosen backbone.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from monai.transforms import (
    Compose, EnsureChannelFirst, RandAdjustContrast, RandAffine,
    RandGaussianNoise, RandScaleIntensity, Resize, ToTensor,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


# Module-level callables (not lambdas) so DataLoader workers can pickle them
# on Windows, where workers are spawned rather than forked.
class RepeatImagenetNorm:
    def __init__(self, channels: int):
        self.c = channels

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] == 1 and self.c > 1:
            x = np.repeat(x, self.c, axis=0)
        return (x - IMAGENET_MEAN[: self.c]) / IMAGENET_STD[: self.c]


class XrvNorm:
    """torchxrayvision expects 1 channel scaled to [-1024, 1024]."""

    def __call__(self, x):
        return (np.asarray(x, dtype=np.float32) * 2048.0 - 1024.0)


class Clip01:
    def __call__(self, x):
        return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def _normalizer(kind: str, channels: int):
    if kind == "imagenet":
        return RepeatImagenetNorm(channels)
    if kind == "xrv":
        return XrvNorm()
    raise ValueError(f"Unknown normalization: {kind}")


def build_transforms(cfg: dict[str, Any], train: bool) -> Compose:
    pp, aug = cfg["preprocess"], cfg["augment"]
    size = pp["image_size"]
    norm = pp["normalization"]
    channels = 1 if norm == "xrv" else pp["in_channels"]

    steps = [
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize((size, size), mode="area"),
    ]
    if train:
        p = aug["prob"]
        s_dev = max(1.0 - aug["scale"][0], aug["scale"][1] - 1.0)
        steps += [
            RandAffine(
                prob=p,
                rotate_range=math.radians(aug["rotate_deg"]),
                scale_range=(s_dev, s_dev),
                translate_range=(aug["translate_frac"] * size, aug["translate_frac"] * size),
                padding_mode="zeros",
            ),
            RandAdjustContrast(prob=p, gamma=tuple(aug["contrast_gamma"])),
            RandScaleIntensity(prob=p, factors=aug["brightness"]),
            RandGaussianNoise(prob=p * 0.5, std=aug["noise_std"]),
            Clip01(),
        ]
        # Deliberately NO RandFlip: see config (laterality matters in CXR).
        assert not aug.get("hflip", False), "Horizontal flip is disabled for chest X-rays"
    steps += [_normalizer(norm, channels), ToTensor(dtype=np.float32, track_meta=False)]
    return Compose(steps)


def denormalize_for_display(t, cfg: dict[str, Any]) -> np.ndarray:
    """Tensor (C,H,W) -> HxW uint8, to visualise augmented samples / overlay Grad-CAM later."""
    x = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if cfg["preprocess"]["normalization"] == "imagenet":
        c = x.shape[0]
        x = x * IMAGENET_STD[:c] + IMAGENET_MEAN[:c]
    else:
        x = (x + 1024.0) / 2048.0
    return (np.clip(x[0], 0, 1) * 255).astype(np.uint8)
