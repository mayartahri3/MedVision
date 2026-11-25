"""Step 3a - pre-resize the 1024x1024 NIH PNGs once (e.g. to 320px).

Decoding 112k full-size PNGs every epoch is the main bottleneck on Kaggle.
A 320px cache is ~5 GB (fits /kaggle/working) and makes epochs several times faster.
Aspect ratio is preserved by padding to square before resizing.

Usage: python scripts/resize_cache.py [--workers 4] [--limit 500]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import load_config  # noqa: E402
from cxr.data.io import pad_to_square, read_xray, to_uint8  # noqa: E402

def process(args: tuple[str, str, int]) -> bool:
    # size travels with the job: globals are not shared with spawned workers (Windows)
    src, dst, size = args
    if Path(dst).exists():
        return True
    try:
        img = pad_to_square(to_uint8(read_xray(src)))
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return cv2.imwrite(dst, img)
    except Exception as e:  # keep going; report at the end
        print(f"FAILED {src}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only N images (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    size = cfg["preprocess"]["cache_size"]

    df = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "splits.csv")
    if args.limit:
        df = df.head(args.limit)
    Path(df["cache_path"].iloc[0]).parent.mkdir(parents=True, exist_ok=True)

    jobs = [(src, dst, size) for src, dst in zip(df["path"], df["cache_path"])]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        ok = sum(ex.map(process, jobs, chunksize=64))
    print(f"Cached {ok:,}/{len(jobs):,} images at {size}px -> {Path(jobs[0][1]).parent}")


if __name__ == "__main__":
    main()
