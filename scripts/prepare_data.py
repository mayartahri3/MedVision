"""Step 2a - build data/processed/splits.csv from the raw NIH ChestX-ray14 files.

Usage (Kaggle):   python scripts/prepare_data.py
Usage (local):    set CXR_DATA_ROOT=D:\\datasets\\nih  &&  python scripts/prepare_data.py

Output columns: Image Index, Patient ID, age, sex, view, path, cache_path,
                <14 NIH labels>, Lung_Opacity, No_Finding, split
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import load_config  # noqa: E402
from cxr.data.labels import parse_findings, positive_weights  # noqa: E402
from cxr.data.split import assign_splits  # noqa: E402


def index_images(raw_root: Path, pattern: str) -> dict[str, str]:
    paths = {p.name: str(p) for p in raw_root.glob(pattern)}
    if not paths:  # flat folder fallback
        paths = {p.name: str(p) for p in raw_root.rglob("*.png")}
    return paths


def read_list(path: Path) -> set[str] | None:
    return set(path.read_text().split()) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw = Path(cfg["paths"]["raw_root"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg["paths"]["cache_root"]) / f"images_{cfg['preprocess']['cache_size']}"

    print(f"Raw data root: {raw}")
    if not (raw / cfg["data"]["metadata_csv"]).exists():
        raise FileNotFoundError(f"{cfg['data']['metadata_csv']} not found in {raw}. "
                                "Set CXR_DATA_ROOT to the folder that contains it.")
    meta = pd.read_csv(raw / cfg["data"]["metadata_csv"])
    print(f"Metadata rows: {len(meta):,}  |  patients: {meta['Patient ID'].nunique():,}")

    images = index_images(raw, cfg["data"]["image_glob"])
    meta["path"] = meta["Image Index"].map(images)
    missing = meta["path"].isna().sum()
    if missing:
        print(f"WARNING: {missing:,} metadata rows have no image file - dropped")
        meta = meta.dropna(subset=["path"])

    meta = meta.rename(columns={"Patient Age": "age", "Patient Gender": "sex",
                                "View Position": "view"})
    meta = meta[meta["view"].isin(cfg["split"]["views"])]
    # A handful of NIH rows have impossible ages (e.g. 412): mark as unknown.
    meta.loc[(meta["age"] <= 0) | (meta["age"] > 100), "age"] = pd.NA

    meta = parse_findings(meta, cfg["labels"]["nih"], cfg["labels"].get("derived"))
    meta["cache_path"] = meta["Image Index"].map(lambda n: str(cache_dir / n))

    test_list = read_list(raw / cfg["data"]["official_test_list"])
    print("Test split:", "official NIH test_list.txt" if test_list else "random 20% of patients")
    meta = assign_splits(meta, test_list, cfg["split"]["val_fraction"], cfg["project"]["seed"])

    keep = ["Image Index", "Patient ID", "Follow-up #", "age", "sex", "view", "path",
            "cache_path", "Finding Labels"]
    label_cols = cfg["labels"]["nih"] + list(cfg["labels"].get("derived", {})) + ["No_Finding"]
    meta[[c for c in keep if c in meta.columns] + label_cols + ["split"]].to_csv(
        out_dir / "splits.csv", index=False)

    train = meta[meta["split"] == "train"]
    pw_labels = cfg["labels"]["nih"] + list(cfg["labels"].get("derived", {}))
    positive_weights(train, pw_labels).round(3).to_csv(out_dir / "pos_weight.csv")

    summary = meta.groupby("split").agg(images=("Image Index", "size"),
                                        patients=("Patient ID", "nunique"))
    print("\n", summary.to_string())
    print(f"\nSaved -> {out_dir / 'splits.csv'} and pos_weight.csv")


if __name__ == "__main__":
    main()
