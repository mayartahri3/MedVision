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

    cfg["paths"]["raw_root"] = _resolve_raw_root(cfg)

    # Outputs default to /kaggle/working; CXR_WORK_DIR relocates all of them
    # (e.g. D:\\cxr_work on Windows) without editing the YAML.
    work = os.getenv("CXR_WORK_DIR")
    if work:
        for sec, key in (("train", "checkpoint_dir"), ("eval", "out_dir"), ("explain", "out_dir")):
            if sec in cfg and str(cfg[sec][key]).startswith("/kaggle/working"):
                cfg[sec][key] = str(Path(work) / Path(cfg[sec][key]).relative_to("/kaggle/working"))

    # Relative paths are resolved against the project root.
    for key in ("processed_dir", "figures_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def _resolve_raw_root(cfg: dict[str, Any]) -> str:
    """Accept a single path or a list of candidates; return the first that holds the metadata CSV."""
    roots = cfg["paths"]["raw_root"]
    roots = [roots] if isinstance(roots, str) else list(roots)
    marker = cfg["data"]["metadata_csv"]
    for r in roots:
        if (Path(r) / marker).exists():
            return str(r)
    # Search /kaggle/input as a last resort (handles future layout changes).
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for depth in ("*", "*/*", "*/*/*"):  # shallow search; avoids walking 112k images
            hit = next(kaggle.glob(f"{depth}/{marker}"), None)
            if hit:
                return str(hit.parent)
    return str(roots[0])  # nothing found: scripts will raise a clear FileNotFoundError


def all_labels(cfg: dict[str, Any]) -> list[str]:
    """Training label order: 14 NIH findings followed by derived labels."""
    return list(cfg["labels"]["nih"]) + list(cfg["labels"].get("derived", {}).keys())
