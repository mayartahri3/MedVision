"""Step 5 - evaluation, calibration and operating points.

1. Predict val + test with the best checkpoint.
2. Fit Platt scaling (scale + bias) per label on VAL (calibration).
3. Choose per-label thresholds on VAL: decision threshold (max F1) and an
   uncertainty band [t_low, t_high] (95% sensitivity / 95% specificity).
4. Report on TEST: AUROC with patient-level 95% CI, AUPRC, sens/spec/PPV,
   ECE before/after calibration, % of cases flagged uncertain, AUROC per view.

Outputs (eval.out_dir): calibration.json  <- used by the API and report generator
    metrics_test.csv, metrics_by_view.csv, test_predictions.csv, eval_report.md,
    roc_focus.png, reliability_focus.png, auroc_ci.png

Usage: python scripts/evaluate.py [--checkpoint path/to/best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.tracking import setup_mlflow  # noqa: E402
from cxr.config import load_config  # noqa: E402
from cxr.data.dataset import make_loaders  # noqa: E402
from cxr.metrics import (apply_platt, ece, fit_platt, operating_points, safe_auc, sigmoid,  # noqa: E402
                         summary_table, threshold_metrics)
from cxr.models.factory import load_checkpoint  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="override loader.batch_size")
    ap.add_argument("--workers", type=int, default=None, help="override loader.num_workers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.batch_size:
        cfg["loader"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["loader"]["num_workers"] = args.workers
    ecfg = cfg["eval"]
    n_boot = args.n_bootstrap if args.n_bootstrap is not None else ecfg["n_bootstrap"]
    out = Path(ecfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or Path(cfg["train"]["checkpoint_dir"]) / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device)
    labels = ckpt["labels"]
    focus = [l for l in cfg["labels"]["focus"] if l in labels]
    nih = [l for l in cfg["labels"]["nih"] if l in labels]
    print(f"Checkpoint epoch {ckpt['epoch']} | val mean AUROC {ckpt['metrics']['val_mean_auroc']:.4f}")

    loaders = make_loaders(cfg, subset=args.subset)
    amp = device.type == "cuda"
    lv, yv = predict(model, loaders["val"], device, amp)
    lt, yt = predict(model, loaders["test"], device, amp)
    test_df = loaders["test"].dataset.df

    # --- calibration + operating points, all fitted on VAL ---------------------------
    calib = {"labels": labels, "model_name": ckpt["model_name"],
             "image_size": ckpt["preprocess"]["image_size"],
             "cache_size": ckpt["preprocess"]["cache_size"],
             "normalization": ckpt["preprocess"]["normalization"], "per_label": {}}
    pv, pt, pt_raw = np.zeros_like(lv), np.zeros_like(lt), sigmoid(lt)
    for i, lab in enumerate(labels):
        a, b = fit_platt(lv[:, i], yv[:, i]) if 0 < yv[:, i].sum() < len(yv) else (1.0, 0.0)
        pv[:, i], pt[:, i] = apply_platt(lv[:, i], a, b), apply_platt(lt[:, i], a, b)
        ops = (operating_points(yv[:, i], pv[:, i], ecfg["sensitivity_target"], ecfg["specificity_target"])
               if 0 < yv[:, i].sum() < len(yv) else {"threshold": 0.5, "t_low": 0.5, "t_high": 0.5})
        calib["per_label"][lab] = {"platt_a": a, "platt_b": b, **{k: round(v, 5) for k, v in ops.items()},
                                   "val_auroc": safe_auc(yv[:, i], pv[:, i])}
    (out / "calibration.json").write_text(json.dumps(calib, indent=2))

    # --- test metrics ---------------------------------------------------------------
    table = summary_table(yt, pt, labels, test_df["Patient ID"].values, n_boot)
    for i, lab in enumerate(labels):
        c = calib["per_label"][lab]
        tm = threshold_metrics(yt[:, i], pt[:, i], c["threshold"])
        for k, v in tm.items():
            table.loc[lab, k] = v
        table.loc[lab, "ece_raw"] = ece(yt[:, i], pt_raw[:, i])
        table.loc[lab, "ece_calibrated"] = ece(yt[:, i], pt[:, i])
        table.loc[lab, "pct_uncertain"] = float(np.mean((pt[:, i] >= c["t_low"]) & (pt[:, i] < c["t_high"])))
    table = table.round(4)
    table.to_csv(out / "metrics_test.csv")

    by_view = {}
    for view in sorted(test_df["view"].dropna().unique()):
        m = (test_df["view"] == view).values
        by_view[view] = {lab: safe_auc(yt[m, i], pt[m, i]) for i, lab in enumerate(labels)}
    by_view = pd.DataFrame(by_view).round(4)
    by_view.to_csv(out / "metrics_by_view.csv")

    preds = pd.DataFrame(pt, columns=[f"p_{l}" for l in labels])
    pd.concat([test_df[["Image Index", "Patient ID", "view"] + labels].reset_index(drop=True), preds],
              axis=1).to_csv(out / "test_predictions.csv", index=False)

    mean14 = float(table.loc[nih, "auroc"].mean())
    _figures(out, labels, focus, yt, pt, pt_raw, table)
    _report(out, table, by_view, mean14, focus, ckpt, n_boot)
    print(table[["n_pos", "auroc"] + [c for c in ("auroc_ci_low", "auroc_ci_high") if c in table]
                + ["sensitivity", "specificity", "ece_calibrated", "pct_uncertain"]].to_string())
    print(f"\nTEST mean AUROC (14 NIH labels): {mean14:.4f}\nOutputs -> {out}")

    print("MLflow tracking:", setup_mlflow(cfg))
    with mlflow.start_run(run_name=f"evaluate-{ckpt['model_name']}"):
        mlflow.log_metric("test_mean_auroc_14", mean14)
        for lab in labels:
            if not np.isnan(table.loc[lab, "auroc"]):
                mlflow.log_metric(f"test_auroc_{lab}", float(table.loc[lab, "auroc"]))
        mlflow.log_artifacts(str(out), artifact_path="eval")


def _figures(out, labels, focus, yt, pt, pt_raw, table):
    idx = {l: labels.index(l) for l in labels}
    # ROC curves for focus labels
    fig, ax = plt.subplots(figsize=(6, 6))
    for lab in focus:
        i = idx[lab]
        if 0 < yt[:, i].sum() < len(yt):
            fpr, tpr, _ = roc_curve(yt[:, i], pt[:, i])
            ax.plot(fpr, tpr, label=f"{lab} (AUROC {table.loc[lab, 'auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - specificity"), ax.set_ylabel("sensitivity"), ax.set_title("ROC - test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "roc_focus.png", dpi=130); plt.close(fig)

    # Reliability diagrams before / after Platt calibration
    fig, axes = plt.subplots(1, len(focus), figsize=(3.6 * len(focus), 3.6), squeeze=False)
    bins = np.linspace(0, 1, 11)
    for ax, lab in zip(axes[0], focus):
        i = idx[lab]
        for p, name, style in ((pt_raw[:, i], "raw", "o--"), (pt[:, i], "calibrated", "s-")):
            b = np.clip(np.digitize(p, bins) - 1, 0, 9)
            xs = [p[b == k].mean() for k in range(10) if (b == k).sum() >= 10]
            ys = [yt[b == k, i].mean() for k in range(10) if (b == k).sum() >= 10]
            ax.plot(xs, ys, style, ms=4, label=name)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(lab, fontsize=9); ax.set_xlabel("predicted"); ax.set_ylabel("observed")
        ax.legend(fontsize=7)
    fig.suptitle("Reliability (test)"); fig.tight_layout()
    fig.savefig(out / "reliability_focus.png", dpi=130); plt.close(fig)

    # AUROC with CI for all labels
    t = table.sort_values("auroc")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(t.index, t["auroc"], color=["#c0504d" if l in focus else "#4f81bd" for l in t.index])
    if "auroc_ci_low" in t:
        ax.errorbar(t["auroc"], range(len(t)), xerr=[t["auroc"] - t["auroc_ci_low"],
                    t["auroc_ci_high"] - t["auroc"]], fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUROC (test, 95% patient-bootstrap CI)")
    fig.tight_layout(); fig.savefig(out / "auroc_ci.png", dpi=130); plt.close(fig)


def _report(out, table, by_view, mean14, focus, ckpt, n_boot):
    cols = ["n_pos", "auroc"] + [c for c in ("auroc_ci_low", "auroc_ci_high") if c in table] + \
           ["auprc", "sensitivity", "specificity", "ppv", "ece_raw", "ece_calibrated", "pct_uncertain"]
    lines = [f"# Evaluation report - {ckpt['model_name']} (epoch {ckpt['epoch']})", "",
             f"**Test mean AUROC (14 NIH labels): {mean14:.4f}**  ",
             f"Validation mean AUROC: {ckpt['metrics']['val_mean_auroc']:.4f}  ",
             f"CIs: {n_boot} patient-level bootstrap resamples." if n_boot else "", "",
             "Platt calibration, thresholds and uncertainty bands were fitted on the validation set only.", "",
             "## Focus labels", "", table.loc[focus, cols].to_markdown(), "",
             "## All labels", "", table[cols].to_markdown(), "",
             "## AUROC by view (PA vs AP)", "", by_view.to_markdown(), "",
             "A large PA/AP gap suggests the model uses acquisition context, not only pathology."]
    (out / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
