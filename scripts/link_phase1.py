"""Connect a Phase 2 session to the saved Phase 1 output.

In the training notebook the Phase 1 output (splits.csv + 320 px cache) is
attached as an *input*, so it lives somewhere under /kaggle/input and its
cache_path column (written as /kaggle/working/cache/...) is stale.
This script finds the files, rewrites cache_path, and copies the CSVs into
data/processed/ so every other script works unchanged.

Usage: python scripts/link_phase1.py                 # auto-detect under /kaggle/input
       python scripts/link_phase1.py --root <dir>    # explicit folder to search
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import load_config  # noqa: E402


def find_one(root: Path, pattern: str, max_depth: int = 7) -> Path | None:
    for d in range(max_depth + 1):
        hits = sorted(root.glob("/".join(["*"] * d + [pattern])))
        if hits:
            return hits[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default="/kaggle/input")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(args.root)
    size = cfg["preprocess"]["cache_size"]

    splits = find_one(root, "splits.csv")
    cache = find_one(root, f"images_{size}")
    if splits is None:
        sys.exit(f"splits.csv not found under {root}. Attach the Phase 1 notebook output "
                 "(Add Input -> Notebooks) or pass --root.")
    print(f"splits.csv : {splits}")
    print(f"image cache: {cache or 'NOT FOUND - will fall back to original 1024px images (slow)'}")

    df = pd.read_csv(splits)
    if cache is not None:
        df["cache_path"] = df["Image Index"].map(lambda n: str(cache / n))
        sample = df["cache_path"].sample(min(200, len(df)), random_state=0)
        found = sample.map(lambda p: Path(p).exists()).mean()
        print(f"cache hit rate on 200 random images: {found:.0%}")
        if found < 0.99:
            print("WARNING: some cached images are missing; those rows will use the originals.")

    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "splits.csv", index=False)
    pw = splits.parent / "pos_weight.csv"
    if pw.exists():
        (out / "pos_weight.csv").write_text(pw.read_text())
    print(f"Linked -> {out / 'splits.csv'} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
