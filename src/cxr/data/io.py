"""Unified image reader for DICOM / PNG / JPG -> float32 grayscale in [0, 1]."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DICOM_EXT = {".dcm", ".dicom"}


def read_xray(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in DICOM_EXT:
        img = _read_dicom(path)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if img.ndim == 3:  # some NIH PNGs are RGBA / RGB
            code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            img = cv2.cvtColor(img, code)
        img = img.astype(np.float32)
    return _minmax(img)


def _read_dicom(path: Path) -> np.ndarray:
    import pydicom
    from pydicom.pixels import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    try:  # apply the windowing the radiologist would see
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr  # MONOCHROME1: bright = low value -> invert
    return arr


def _minmax(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img, dtype=np.float32)


def to_uint8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 255).round().astype(np.uint8)
