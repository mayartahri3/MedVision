"""Step 6 - Grad-CAM: localisation check (pointing game) + overlay gallery.

NIH provides ~1,000 radiologist-drawn boxes (BBox_List_2017.csv) for 8 findings.
The model never saw them in training, so they let us measure whether Grad-CAM
points at the right place:

  hit      = the CAM's hottest pixel falls inside the box (+ tolerance)
  energy   = share of total CAM mass that lies inside the box
  baseline = same metrics for a trivial "always point at the image centre" guess

Usage: python scripts/gradcam_eval.py [--checkpoint best.pt] [--split test|all] [--limit N]
Outputs (explain.out_dir): pointing_game.csv, pointing_summary.csv,
                            gradcam_gallery.png, explain_report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import load_config  # noqa: E402
from cxr.explain.gradcam import overlay  # noqa: E402
from cxr.inference import Predictor  # noqa: E402

BBOX_NAME_FIX = {"Infiltrate": "Infiltration"}


def load_boxes(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.iloc[:, :6].copy()
    df.columns = ["Image Index", "label", "x", "y", "w", "h"]
    df["label"] = df["label"].replace(BBOX_NAME_FIX)
    return df


def to_model_coords(box, orig_hw, size):
    """Map a box from original-image pixels to model-input pixels (pad-to-square + resize)."""
    h, w = orig_hw
    s = max(h, w); top, left = (s - h) // 2, (s - w) // 2
    k = size / s
    x, y, bw, bh = box
    return (x + left) * k, (y + top) * k, bw * k, bh * k, k


def in_box(px, py, bx, by, bw, bh, tol):
    return (bx - tol <= px <= bx + bw + tol) and (by - tol <= py <= by + bh + tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--split", default="test", choices=["test", "all"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    xcfg = cfg["explain"]
    out = Path(xcfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint or Path(cfg["train"]["checkpoint_dir"]) / "best.pt"
    calib = args.calibration or Path(cfg["eval"]["out_dir"]) / "calibration.json"
    pred = Predictor(ckpt, calib if Path(calib).exists() else None)
    size = pred.pp["image_size"]

    boxes = load_boxes(Path(cfg["paths"]["raw_root"]) / cfg["data"]["bbox_csv"])
    splits = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "splits.csv",
                         usecols=["Image Index", "path", "split"])
    boxes = boxes.merge(splits, on="Image Index", how="inner")
    print("Boxes per split:", boxes["split"].value_counts().to_dict())
    if args.split == "test":
        boxes = boxes[boxes["split"] == "test"]
    boxes = boxes[boxes["label"].isin(pred.labels)]
    if args.limit:
        boxes = boxes.head(args.limit)

    rows, gallery = [], {}
    for img_name, grp in boxes.groupby("Image Index"):
        path = grp["path"].iloc[0]
        orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        disp, x = pred.preprocess(path)
        labs = grp["label"].unique().tolist()
        logits, cams = pred.cam(x, [pred.labels.index(l) for l in labs])
        for _, r in grp.iterrows():
            cam = cams[labs.index(r["label"])]
            bx, by, bw, bh, k = to_model_coords((r.x, r.y, r.w, r.h), orig.shape, size)
            tol = xcfg["pointing_tolerance_px"] * k
            py, px = np.unravel_index(np.argmax(cam), cam.shape)
            c = size / 2
            mask = np.zeros_like(cam, bool)
            mask[int(max(by, 0)):int(min(by + bh, size)) + 1, int(max(bx, 0)):int(min(bx + bw, size)) + 1] = True
            rows.append({"Image Index": img_name, "label": r["label"],
                         "prob": float(1 / (1 + np.exp(-logits[pred.labels.index(r['label'])]))),
                         "hit": in_box(px, py, bx, by, bw, bh, tol),
                         "energy_in_box": float(cam[mask].sum() / max(cam.sum(), 1e-8)),
                         "box_area_frac": float(mask.mean()),
                         "center_hit": in_box(c, c, bx, by, bw, bh, tol)})
            if len(gallery.get(r["label"], [])) < 2:
                gallery.setdefault(r["label"], []).append(
                    (overlay(disp, cam), (bx, by, bw, bh), (px, py), disp.shape[0] / size, rows[-1]["hit"]))

    if not rows:
        sys.exit("No bounding boxes matched the selected split.")
    res = pd.DataFrame(rows)
    res.to_csv(out / "pointing_game.csv", index=False)
    summ = res.groupby("label").agg(n=("hit", "size"), pointing_acc=("hit", "mean"),
                                    centre_baseline=("center_hit", "mean"),
                                    energy_in_box=("energy_in_box", "mean"),
                                    box_area=("box_area_frac", "mean"))
    summ.loc["ALL"] = [len(res), res.hit.mean(), res.center_hit.mean(),
                       res.energy_in_box.mean(), res.box_area_frac.mean()]
    summ = summ.round(3)
    summ.to_csv(out / "pointing_summary.csv")
    print(summ.to_string())

    _gallery(gallery, out / "gradcam_gallery.png")
    (out / "explain_report.md").write_text("\n".join([
        "# Grad-CAM localisation (NIH bounding boxes)", "",
        f"Split: {args.split}; boxes evaluated: {len(res)}; tolerance: "
        f"{xcfg['pointing_tolerance_px']} px (1024-px space).", "",
        summ.to_markdown(), "",
        "- **pointing_acc**: hottest CAM pixel inside the box.",
        "- **centre_baseline**: same test for a fixed guess at the image centre. "
        "Grad-CAM is only informative where it clearly beats this.",
        "- **energy_in_box vs box_area**: if energy is not well above the box's share of the "
        "image, the heatmap is diffuse rather than localised.",
        "", "Grad-CAM shows where the model looked, not proof that its reasoning is correct."]),
        encoding="utf-8")
    print(f"Outputs -> {out}")


def _gallery(gallery, path):
    items = [(lab, it) for lab, its in sorted(gallery.items()) for it in its]
    if not items:
        return
    cols = 4; rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (lab, (img, (bx, by, bw, bh), (px, py), s, hit)) in zip(axes.ravel(), items):
        ax.imshow(img)
        ax.add_patch(patches.Rectangle((bx * s, by * s), bw * s, bh * s, fill=False, ec="lime", lw=1.5))
        ax.plot(px * s, py * s, "w+", ms=12, mew=2)
        ax.set_title(f"{lab} - {'hit' if hit else 'miss'}", fontsize=9)
    fig.suptitle("Grad-CAM (green = radiologist box, + = hottest point)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
