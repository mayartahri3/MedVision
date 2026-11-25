"""Grad-CAM without hooks.

torchvision's DenseNet applies an *in-place* ReLU right after `features`,
which breaks hook-based Grad-CAM implementations in subtle ways. Instead we
re-run the forward pass explicitly, keep the last feature map as a tensor in
the graph, and take gradients with torch.autograd.grad.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_with_features(model: nn.Module, name: str, x: torch.Tensor):
    """Return (logits, A) where A is the last conv feature map used for Grad-CAM."""
    if name == "densenet121":
        A = F.relu(model.features(x))
        logits = model.classifier(torch.flatten(F.adaptive_avg_pool2d(A, 1), 1))
    elif name == "resnet50":
        m = model
        h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        A = m.layer4(m.layer3(m.layer2(m.layer1(h))))
        logits = m.fc(torch.flatten(m.avgpool(A), 1))
    elif name in ("efficientnet_b0", "efficientnet_v2_s"):
        A = model.features(x)
        logits = model.classifier(torch.flatten(model.avgpool(A), 1))
    elif name == "convnext_tiny":
        A = model.features(x)
        logits = model.classifier(model.avgpool(A))  # LayerNorm2d -> Flatten -> head
    else:
        raise ValueError(name)
    return logits, A


class GradCAM:
    def __init__(self, model: nn.Module, model_name: str):
        self.model = model.eval()
        self.name = model_name

    def __call__(self, x: torch.Tensor, class_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """x: (1, C, H, W). Returns (logits[num_labels], cams[len(class_indices), H, W] in [0,1])."""
        x = x.requires_grad_(False)
        with torch.enable_grad():
            logits, A = forward_with_features(self.model, self.name, x)
            cams = []
            for k, c in enumerate(class_indices):
                (g,) = torch.autograd.grad(logits[0, c], A, retain_graph=k < len(class_indices) - 1)
                w = g.mean(dim=(2, 3), keepdim=True)                  # channel importance
                cam = F.relu((w * A).sum(dim=1, keepdim=True))         # (1,1,h,w)
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                cam = cam - cam.min()
                cams.append((cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy())
        return logits[0].detach().cpu().numpy(), np.stack(cams) if cams else np.zeros((0,) + x.shape[-2:])


def overlay(gray_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0,1] heatmap onto a grayscale image -> RGB uint8."""
    cam = cv2.resize(cam.astype(np.float32), gray_uint8.shape[::-1])
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    base = np.repeat(gray_uint8[:, :, None], 3, axis=2)
    return (base * (1 - alpha) + heat * alpha).astype(np.uint8)
