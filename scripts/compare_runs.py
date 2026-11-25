"""Compare every evaluated model in MLflow -> leaderboard + charts (logged as a run).

The model is SELECTED BY VALIDATION AUROC, not test AUROC: picking the best test
score would leak the test set into model selection. Test numbers are reported
for every model so the comparison stays transparent.

Usage: python scripts/compare_runs.py            (needs MLFLOW_TRACKING_URI or the local mlflow.db)
Outputs: <work>/comparison/  leaderboard.csv / .md, per_label_auroc.png,
         leaderboard.png, val_curves.png, auroc_vs_size.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import all_labels, load_config  # noqa: E402
from cxr.tracking import base_tags, setup_mlflow  # noqa: E402

COLS = {  # MLflow column -> leaderboard column
    "tags.experiment": "experiment", "tags.model": "model", "tags.image_size": "img",
    "params.n_params_m": "params_M", "metrics.best_epoch": "best_epoch", "metrics.train_minutes": "train_min",
    "metrics.best_val_mean_auroc": "val_auroc", "metrics.test_mean_auroc_14": "test_auroc",
    "metrics.test_mean_auroc_14_no_tta": "test_auroc_noTTA", "metrics.test_mean_auroc_focus": "test_auroc_focus",
    "metrics.test_mean_auprc_14": "test_auprc", "metrics.test_mean_ece_calibrated": "ece_cal",
    "metrics.uncertain_per_image_focus": "uncertain_per_img", "metrics.test_mean_auroc_14_AP": "auroc_AP",
    "metrics.test_mean_auroc_14_PA": "auroc_PA", "metrics.explain_pointing_acc": "gradcam_hit",
    "metrics.explain_centre_baseline": "centre_baseline", "metrics.val_mean_auroc_14_calibrated": "val_auroc_cal",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-log", action="store_true", help="do not create the comparison run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_mlflow(cfg)
    exp = cfg["mlflow"]["experiment"]

    runs = mlflow.search_runs(experiment_names=[exp], filter_string="metrics.test_mean_auroc_14 > 0")
    if "tags.smoke_test" in runs:
        runs = runs[runs["tags.smoke_test"].ne("true")]
    if "tags.type" in runs:
        runs = runs[runs["tags.type"].ne("comparison")]
    if runs.empty:
        sys.exit("No evaluated runs found.")
    # keep the latest run per experiment name
    runs = runs.sort_values("start_time").groupby("tags.experiment", as_index=False).tail(1)

    lb = pd.DataFrame({new: runs[old] if old in runs else np.nan for old, new in COLS.items()})
    lb["run_id"] = runs["run_id"].values
    lb["val_for_selection"] = lb["val_auroc_cal"].fillna(lb["val_auroc"])
    lb = lb.sort_values("val_for_selection", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank_by_val", range(1, len(lb) + 1))
    num = lb.select_dtypes("number").columns
    lb[num] = lb[num].astype(float).round(4)
    selected = lb.iloc[0]

    out = Path(cfg["eval"]["out_dir"]).parent / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out / "leaderboard.csv", index=False)
    show = ["rank_by_val", "experiment", "model", "img", "params_M", "val_for_selection", "test_auroc",
            "test_auroc_noTTA", "test_auroc_focus", "test_auprc", "ece_cal", "uncertain_per_img",
            "gradcam_hit", "centre_baseline", "train_min"]
    md = ["# Model comparison - NIH ChestX-ray14 (official test set)", "",
          f"**Selected model (highest validation AUROC): `{selected['experiment']}`** - "
          f"test mean AUROC {selected['test_auroc']:.4f}", "",
          "Selection uses validation only; test columns are reported for transparency.", "",
          lb[show].to_markdown(index=False)]

    labels = all_labels(cfg)
    per_label = pd.DataFrame({r["experiment"]: {l: runs.loc[runs.run_id == r["run_id"], f"metrics.test_auroc_{l}"].iloc[0]
                                                if f"metrics.test_auroc_{l}" in runs else np.nan for l in labels}
                              for _, r in lb.iterrows()}).astype(float)
    per_label.round(4).to_csv(out / "per_label_auroc.csv")
    md += ["", "## Test AUROC per label", "", per_label.round(3).to_markdown()]
    (out / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")

    _charts(out, lb, per_label, runs, cfg)
    print("\n".join(md[:3]))
    print(lb[show].to_string(index=False))

    if not args.no_log:
        with mlflow.start_run(run_name="model-comparison"):
            tags = base_tags(cfg)
            tags.update({"type": "comparison", "experiment": "model-comparison", "model": "-",
                         "selected_experiment": str(selected["experiment"]),
                         "selected_run_id": str(selected["run_id"]),
                         "mlflow.note.content": "Leaderboard of all evaluated models (selection by validation)."})
            mlflow.set_tags(tags)
            mlflow.log_metrics({"n_models": len(lb), "selected_test_auroc": float(selected["test_auroc"]),
                                "selected_val_auroc": float(selected["val_for_selection"])})
            mlflow.log_artifacts(str(out), artifact_path="comparison")
    print(f"\nOutputs -> {out}")


def _charts(out: Path, lb: pd.DataFrame, per_label: pd.DataFrame, runs: pd.DataFrame, cfg) -> None:
    order = lb["experiment"].tolist()
    # 1. leaderboard bars: val vs test
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(lb) + 1.5))
    y = np.arange(len(lb))
    ax.barh(y - 0.2, lb["val_for_selection"], 0.4, label="validation", color="#9bbb59")
    ax.barh(y + 0.2, lb["test_auroc"], 0.4, label="test", color="#4f81bd")
    for i, v in enumerate(lb["test_auroc"]):
        ax.text(v + 0.002, i + 0.2, f"{v:.3f}", va="center", fontsize=8)
    lo = float(np.nanmin(lb[["val_for_selection", "test_auroc"]].values))
    ax.set_yticks(y, order); ax.invert_yaxis(); ax.set_xlim(max(0.0, lo - 0.03), min(1.0, lo + 0.15))
    ax.set_xlabel("mean AUROC (14 NIH labels)"); ax.legend(loc="lower right")
    ax.set_title("Models ranked by validation AUROC")
    fig.tight_layout(); fig.savefig(out / "leaderboard.png", dpi=130); plt.close(fig)

    # 2. per-label heatmap
    fig, ax = plt.subplots(figsize=(1.3 * len(per_label.columns) + 4, 7))
    sns.heatmap(per_label[order], annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.65, vmax=0.95, ax=ax,
                cbar_kws={"label": "test AUROC"})
    ax.set_title("Test AUROC per label and model"); plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out / "per_label_auroc.png", dpi=130); plt.close(fig)

    # 3. validation curves from the metric history
    client = mlflow.tracking.MlflowClient()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for _, r in lb.iterrows():
        try:
            hist = client.get_metric_history(r["run_id"], "val_mean_auroc")
        except Exception:
            hist = []
        if hist:
            h = sorted(hist, key=lambda m: m.step)
            ax.plot([m.step for m in h], [m.value for m in h], "o-", ms=3, label=r["experiment"])
    ax.set_xlabel("epoch"); ax.set_ylabel("val mean AUROC"); ax.set_title("Validation AUROC during training")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "val_curves.png", dpi=130); plt.close(fig)

    # 4. accuracy vs model size
    d = lb.dropna(subset=["params_M"])
    if len(d):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(d["params_M"], d["test_auroc"], s=60, color="#c0504d")
        for _, r in d.iterrows():
            ax.annotate(r["experiment"], (r["params_M"], r["test_auroc"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("parameters (millions)"); ax.set_ylabel("test mean AUROC")
        ax.set_title("Accuracy vs model size")
        fig.tight_layout(); fig.savefig(out / "auroc_vs_size.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
