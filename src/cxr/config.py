"""Config loading with optional environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Env overrides make the same config work on Kaggle, Colab and Windows.
    if os.getenv("CXR_DATA_ROOT"):
        cfg["paths"]["raw_root"] = os.environ["CXR_DATA_ROOT"]
    if os.getenv("CXR_CACHE_ROOT"):
        cfg["paths"]["cache_root"] = os.environ["CXR_CACHE_ROOT"]

    # Relative paths are resolved against the project root.
    for key in ("processed_dir", "figures_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def all_labels(cfg: dict[str, Any]) -> list[str]:
    """Training label order: 14 NIH findings followed by derived labels."""
    return list(cfg["labels"]["nih"]) + list(cfg["labels"].get("derived", {}).keys())
