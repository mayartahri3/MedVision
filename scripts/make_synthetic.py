"""Create a tiny fake NIH-style dataset to test the pipeline without the 45 GB download.

Usage:  python scripts/make_synthetic.py --out data/synthetic --n 400
        CXR_DATA_ROOT=data/synthetic CXR_CACHE_ROOT=data/synthetic_cache python scripts/prepare_data.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
          "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
          "Pleural_Thickening", "Hernia"]


def fake_cxr(rng, size=512) -> np.ndarray:
    img = np.full((size, size), 40, np.uint8)
    c = size // 2
    cv2.ellipse(img, (c - size // 5, c), (size // 7, size // 3), 0, 0, 360, 110, -1)  # lungs
    cv2.ellipse(img, (c + size // 5, c), (size // 7, size // 3), 0, 0, 360, 110, -1)
    cv2.ellipse(img, (c + size // 16, c + size // 8), (size // 8, size // 9), 0, 0, 360, 190, -1)  # heart (patient's left)
    cv2.rectangle(img, (c - 8, 0), (c + 8, size), 210, -1)  # spine
    img = cv2.GaussianBlur(img, (21, 21), 0)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.out)
    img_dir = out / "images_001" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows, n_pat = [], max(args.n // 4, 10)
    for i in range(args.n):
        pid = int(rng.integers(1, n_pat + 1))
        k = rng.choice([0, 1, 2], p=[0.55, 0.3, 0.15])
        finding = "|".join(rng.choice(LABELS, size=k, replace=False)) if k else "No Finding"
        name = f"{pid:08d}_{i:03d}.png"
        cv2.imwrite(str(img_dir / name), fake_cxr(rng))
        rows.append({"Image Index": name, "Finding Labels": finding, "Follow-up #": i,
                     "Patient ID": pid, "Patient Age": int(rng.integers(1, 95)),
                     "Patient Gender": rng.choice(["M", "F"]),
                     "View Position": rng.choice(["PA", "AP"], p=[0.6, 0.4])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "Data_Entry_2017.csv", index=False)
    test_pats = set(rng.choice(df["Patient ID"].unique(), size=n_pat // 5, replace=False))
    (out / "test_list.txt").write_text("\n".join(df[df["Patient ID"].isin(test_pats)]["Image Index"]))
    print(f"Synthetic dataset: {len(df)} images, {df['Patient ID'].nunique()} patients -> {out}")


if __name__ == "__main__":
    main()
