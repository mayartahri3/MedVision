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


def pad_to_square(img: np.ndarray, value: float = 0) -> np.ndarray:
    """Pad (centered) so that H == W. Same geometry is used for the cache and at inference."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=value)


def load_like_cache(path: str | Path, cache_size: int) -> np.ndarray:
    """Read any raw X-ray exactly the way resize_cache.py prepared training images.

    Used at inference (API, Grad-CAM on raw files) so that the model sees the
    same distribution it was trained on: [0,1] -> uint8 -> pad -> resize -> [0,1].
    """
    img = pad_to_square(to_uint8(read_xray(path)))
    img = cv2.resize(img, (cache_size, cache_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0
