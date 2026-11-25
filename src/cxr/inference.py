"""Single-image inference: calibrated probabilities, 3-way status and Grad-CAM.

This is the object the API (Step 9) and the report generator (Step 7) use.
Its output is plain, JSON-serialisable data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cxr.data.io import load_like_cache
from cxr.data.transforms import build_transforms
from cxr.explain.gradcam import GradCAM, overlay
from cxr.metrics import apply_platt, sigmoid
from cxr.models.factory import load_checkpoint

DISCLAIMER = ("Research prototype - not a medical device. Findings are AI-generated "
              "suggestions and must be reviewed by a qualified radiologist.")


class Predictor:
    def __init__(self, checkpoint: str | Path, calibration: str | Path | None = None,
                 device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.model_name = ckpt["model_name"]
        self.labels: list[str] = ckpt["labels"]
        self.pp = ckpt["preprocess"]
        self.calib = json.loads(Path(calibration).read_text())["per_label"] if calibration else None
        self.transform = build_transforms({"preprocess": self.pp, "augment": {}}, train=False)
        self.cam = GradCAM(self.model, self.model_name)

    def preprocess(self, path: str | Path) -> tuple[np.ndarray, torch.Tensor]:
        img01 = load_like_cache(path, self.pp["cache_size"])       # same geometry as training
        x = self.transform(img01).unsqueeze(0).to(self.device)
        disp = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        return disp, x

    def _status(self, label: str, p: float) -> tuple[str, dict[str, float]]:
        if not self.calib:
            return ("positive" if p >= 0.5 else "negative"), {"threshold": 0.5}
        c = self.calib[label]
        status = "negative" if p < c["t_low"] else "positive" if p >= c["t_high"] else "uncertain"
        return status, {k: c[k] for k in ("threshold", "t_low", "t_high")}

    def predict(self, path: str | Path, explain: str | list[str] = "flagged") -> dict[str, Any]:
        """explain: 'flagged' (positive + uncertain), 'all', 'none', or a list of labels."""
        disp, x = self.preprocess(path)
        with torch.no_grad():
            logits = self.model(x)[0].float().cpu().numpy()

        findings = []
        for i, lab in enumerate(self.labels):
            raw = float(sigmoid(logits[i]))
            p = float(apply_platt(logits[i], self.calib[lab]["platt_a"], self.calib[lab]["platt_b"])) \
                if self.calib else raw
            status, ops = self._status(lab, p)
            findings.append({"label": lab, "probability": round(p, 4), "raw_probability": round(raw, 4),
                             "status": status, **{k: round(v, 4) for k, v in ops.items()}})
        findings.sort(key=lambda f: f["probability"], reverse=True)

        if explain == "all":
            targets = self.labels
        elif explain == "none":
            targets = []
        elif explain == "flagged":
            targets = [f["label"] for f in findings if f["status"] != "negative"]
        else:
            targets = list(explain)
        heatmaps = {}
        if targets:
            _, cams = self.cam(x, [self.labels.index(t) for t in targets])
            heatmaps = {t: cams[k] for k, t in enumerate(targets)}

        return {"findings": findings, "heatmaps": heatmaps, "image": disp,
                "model": self.model_name, "calibrated": self.calib is not None,
                "disclaimer": DISCLAIMER}

    @staticmethod
    def overlay(result: dict[str, Any], label: str, alpha: float = 0.4) -> np.ndarray:
        return overlay(result["image"], result["heatmaps"][label], alpha)
