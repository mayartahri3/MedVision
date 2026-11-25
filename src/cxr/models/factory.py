"""Model construction and checkpoint I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_labels: int, pretrained: bool = True,
                dropout: float = 0.0) -> nn.Module:
    if name == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier.in_features, num_labels))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_labels))
    elif name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_labels))
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        # classifier = [LayerNorm2d, Flatten, Linear] -> keep the norm, swap the head
        m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[2].in_features, num_labels))
    elif name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_labels))
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional block: the layer Grad-CAM reads activations from."""
    if name == "densenet121":
        return model.features          # output of norm5, 7x7 at 224 px
    if name in ("efficientnet_b0", "efficientnet_v2_s", "convnext_tiny"):
        return model.features[-1]
    if name == "resnet50":
        return model.layer4
    raise ValueError(name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: str | Path, model: nn.Module, cfg: dict[str, Any],
                    labels: list[str], epoch: int, metrics: dict[str, float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": cfg["model"]["name"],
                "experiment": cfg.get("experiment", {}).get("name", "default"),
                "dropout": cfg["model"]["dropout"], "labels": labels, "epoch": epoch,
                "metrics": metrics, "preprocess": cfg["preprocess"]}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["labels"]), pretrained=False,
                        dropout=ckpt.get("dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
