"""Step 2b - exploratory data analysis. Writes figures + a markdown summary.

Usage: python scripts/eda.py
Outputs (reports/figures/): label_prevalence.png, cooccurrence.png,
    labels_per_image.png, demographics.png, views.png, split_balance.png,
    sample_grid.png, eda_summary.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import load_config  # noqa: E402
from cxr.data.io import read_xray  # noqa: E402

sns.set_theme(style="whitegrid", font_scale=0.9)


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "splits.csv")
    nih = cfg["labels"]["nih"]
    derived = list(cfg["labels"].get("derived", {}))
    labels = nih + derived
    lines = ["# EDA summary - NIH ChestX-ray14", ""]

    # 1. Label prevalence (class imbalance)
    prev = df[labels + ["No_Finding"]].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#c0504d" if l in cfg["labels"]["focus"] else "#4f81bd" for l in prev.index]
    prev.plot.barh(ax=ax, color=colors)
    ax.set_xlabel("% of images"), ax.set_title("Label prevalence (red = focus labels)")
    for i, v in enumerate(prev.values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    save(fig, out, "label_prevalence.png")
    lines += ["## Prevalence (% of images)", "", prev.round(2).to_frame("pct").to_markdown(), ""]

    # 2. Co-occurrence (conditional: P(col | row))
    m = df[nih].values.astype(float)
    co = (m.T @ m) / np.maximum(m.sum(0)[:, None], 1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(pd.DataFrame(co, nih, nih), cmap="mako_r", vmin=0, vmax=0.6, ax=ax,
                annot=True, fmt=".2f", annot_kws={"size": 6}, cbar_kws={"label": "P(col | row)"})
    ax.set_title("Finding co-occurrence")
    save(fig, out, "cooccurrence.png")

    # 3. Labels per image
    n_lab = df[nih].sum(axis=1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    n_lab.value_counts().sort_index().plot.bar(ax=ax, color="#4f81bd")
    ax.set_xlabel("# findings per image"), ax.set_ylabel("images")
    save(fig, out, "labels_per_image.png")
    lines += [f"- Multi-label images (>=2 findings): {(n_lab >= 2).mean():.1%}"]

    # 4. Demographics
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    sns.histplot(df, x="age", hue="sex", bins=40, ax=axes[0], element="step")
    axes[0].set_title("Age by sex")
    per_pat = df.groupby("Patient ID").size()
    sns.histplot(per_pat.clip(upper=50), bins=50, ax=axes[1], color="#9bbb59")
    axes[1].set_title("Images per patient (clipped at 50)"), axes[1].set_yscale("log")
    save(fig, out, "demographics.png")
    lines += [f"- Patients: {df['Patient ID'].nunique():,}; median images/patient: "
              f"{per_pat.median():.0f}; max: {per_pat.max()}  -> patient-level split is mandatory",
              f"- Sex: {df['sex'].value_counts(normalize=True).round(3).to_dict()}"]

    # 5. View position vs findings (AP films are sicker / bedside patients)
    fig, ax = plt.subplots(figsize=(8, 4))
    vp = df.groupby("view")[labels].mean().T * 100
    vp.plot.bar(ax=ax)
    ax.set_ylabel("% positive"), ax.set_title("Prevalence by view position (PA vs AP)")
    save(fig, out, "views.png")
    lines += [f"- Views: {df['view'].value_counts().to_dict()} - AP is a confounder "
              "(bedside, sicker patients); report metrics per view in Step 5."]

    # 6. Split balance
    sb = df.groupby("split")[labels].mean().T * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    sb[["train", "val", "test"]].plot.bar(ax=ax)
    ax.set_ylabel("% positive"), ax.set_title("Label prevalence per split")
    save(fig, out, "split_balance.png")
    sizes = df.groupby("split").agg(images=("Image Index", "size"),
                                    patients=("Patient ID", "nunique"))
    lines += ["", "## Splits", "", sizes.to_markdown(), "",
              "Note: the official NIH test set is enriched in positives "
              "(follow-up studies of sick patients) - expect test prevalence > train.", ""]

    # 7. Sample grid: 2 examples per focus label
    focus = cfg["labels"]["focus"]
    fig, axes = plt.subplots(2, len(focus), figsize=(3 * len(focus), 6.4))
    rng = np.random.default_rng(cfg["project"]["seed"])
    for j, lab in enumerate(focus):
        pos = df[df[lab] == 1]
        picks = pos.iloc[rng.choice(len(pos), size=min(2, len(pos)), replace=False)] if len(pos) else []
        for i in range(2):
            ax = axes[i, j]
            ax.axis("off")
            if i < len(picks):
                row = picks.iloc[i]
                p = row["cache_path"] if Path(str(row["cache_path"])).exists() else row["path"]
                ax.imshow(read_xray(p), cmap="gray")
                ax.set_title(f"{lab}\n{row['Finding Labels'][:28]}", fontsize=8)
    save(fig, out, "sample_grid.png")

    lines += ["## Label-quality caveat", "",
              "NIH labels were text-mined from reports (NLP), not read by radiologists; "
              "some labels are wrong. Treat ceiling performance with caution and inspect "
              "errors visually in Step 5."]
    (out / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA written to {out}")


if __name__ == "__main__":
    main()
