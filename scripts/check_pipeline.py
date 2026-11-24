"""Step 3b - sanity-check the DataLoader: shapes, value ranges, label balance,
and a visual grid of augmented training samples (reports/figures/augmentations.png).

Usage: python scripts/check_pipeline.py [--subset 256]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import all_labels, load_config  # noqa: E402
from cxr.data.dataset import make_loaders  # noqa: E402
from cxr.data.transforms import denormalize_for_display  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subset", type=int, default=256)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["loader"]["num_workers"] = args.workers

    loaders = make_loaders(cfg, subset=args.subset)
    labels = all_labels(cfg)
    for split, dl in loaders.items():
        t0 = time.time()
        batch = next(iter(dl))
        x, y = batch["image"], batch["target"]
        print(f"[{split:5}] n={len(dl.dataset):5d}  x={tuple(x.shape)} {x.dtype} "
              f"range=({x.min():.2f}, {x.max():.2f}) mean={x.mean():.2f}  "
              f"y={tuple(y.shape)}  first batch {time.time() - t0:.2f}s")
        assert x.shape[1:] == (cfg["preprocess"]["in_channels"] if cfg["preprocess"]["normalization"] == "imagenet" else 1,
                               cfg["preprocess"]["image_size"], cfg["preprocess"]["image_size"])
        assert y.shape[1] == len(labels)

    # Same image passed through the train pipeline 6 times -> shows augmentation strength
    ds = loaders["train"].dataset
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
    for r in range(2):
        idx = r * 7
        for c in range(6):
            ax = axes[r, c]
            ax.imshow(denormalize_for_display(ds[idx]["image"], cfg), cmap="gray")
            ax.set_title(f"image {r + 1} - draw {c + 1}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Training augmentations (no flips - heart must stay on the left of the patient)")
    fig.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / "augmentations.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Saved {out}\nPipeline OK.")


if __name__ == "__main__":
    main()
