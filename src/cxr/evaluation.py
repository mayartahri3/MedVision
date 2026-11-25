"""Evaluation core shared by evaluate.py (one model) and ensemble.py (several).

Input: logits on VAL and TEST. Everything that is *fitted* (Platt calibration,
uncertainty band) uses VAL only; everything that is *reported* uses TEST.
If an MLflow run is active, metrics and all outputs are logged to it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from cxr.metrics import (apply_platt, band_thresholds, ece, fit_platt, safe_auc,  # noqa: E402
                         sigmoid, status_counts, summary_table, threshold_metrics)


# ----------------------------------------------------------------------------- TTA
def _zoom(x: torch.Tensor, scale: float, dx: float = 0.0, dy: float = 0.0) -> torch.Tensor:
    theta = torch.tensor([[1 / scale, 0, dx], [0, 1 / scale, dy]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta.unsqueeze(0).expand(len(x), -1, -1), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def tta_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Average of 4 mild views. No flips (laterality matters in chest X-rays)."""
    views = [x, _zoom(x, 1.08), _zoom(x, 0.95), _zoom(x, 1.0, dx=0.03, dy=0.03)]
    return torch.stack([model(v) for v in views]).mean(0)


# ----------------------------------------------------------------------------- core
def _band_targets(cfg: dict[str, Any], label: str) -> tuple[float, float]:
    b = cfg["eval"]["band"]
    o = (b.get("per_label") or {}).get(label, {})
    return o.get("sensitivity", b["sensitivity"]), o.get("specificity", b["specificity"])


def evaluate_logits(lv: np.ndarray, yv: np.ndarray, lt: np.ndarray, yt: np.ndarray,
                    test_df: pd.DataFrame, labels: list[str], cfg: dict[str, Any], out: Path,
                    meta: dict[str, Any], val_ids=None, extra_metrics: dict[str, float] | None = None,
                    log_to_mlflow: bool = True) -> dict[str, Any]:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ecfg = cfg["eval"]
    focus = [l for l in cfg["labels"]["focus"] if l in labels]
    nih = [l for l in cfg["labels"]["nih"] if l in labels]
    ok = lambda y: 0 < y.sum() < len(y)  # noqa: E731

    # ---- fit on VAL: calibration + band ---------------------------------------
    calib = {"labels": labels, **meta, "band_policy": ecfg["band"], "per_label": {}}
    pv, pt = np.zeros_like(lv), np.zeros_like(lt)
    for i, lab in enumerate(labels):
        a, b = fit_platt(lv[:, i], yv[:, i]) if ok(yv[:, i]) else (1.0, 0.0)
        pv[:, i], pt[:, i] = apply_platt(lv[:, i], a, b), apply_platt(lt[:, i], a, b)
        sens, spec = _band_targets(cfg, lab)
        band = band_thresholds(yv[:, i], pv[:, i], sens, spec) if ok(yv[:, i]) else {"t_low": .5, "t_high": .5}
        calib["per_label"][lab] = {"platt_a": a, "platt_b": b, "threshold": round(band["t_high"], 5),
                                   "t_low": round(band["t_low"], 5), "t_high": round(band["t_high"], 5),
                                   "target_sensitivity": sens, "target_specificity": spec,
                                   "val_auroc": safe_auc(yv[:, i], pv[:, i])}
    (out / "calibration.json").write_text(json.dumps(calib, indent=2))

    # ---- band-width sweep on VAL (justifies the chosen targets) ---------------
    sweep = []
    for tgt in ecfg["sweep_targets"]:
        row = {"target": tgt}
        for i, lab in enumerate(labels):
            if ok(yv[:, i]):
                bt = band_thresholds(yv[:, i], pv[:, i], tgt, tgt)
                row[lab] = status_counts(pv[:, i], bt["t_low"], bt["t_high"])["pct_uncertain"]
        sweep.append(row)
    sweep = pd.DataFrame(sweep).set_index("target").round(3)
    sweep.to_csv(out / "band_sweep_val.csv")

    # ---- report on TEST -------------------------------------------------------
    pt_raw = sigmoid(lt)
    table = summary_table(yt, pt, labels, test_df["Patient ID"].values, ecfg["n_bootstrap"])
    status = np.zeros(pt.shape, dtype=np.int8)        # 0 neg, 1 uncertain, 2 pos
    for i, lab in enumerate(labels):
        c = calib["per_label"][lab]
        pos = threshold_metrics(yt[:, i], pt[:, i], c["t_high"])
        rule_out = threshold_metrics(yt[:, i], pt[:, i], c["t_low"])
        table.loc[lab, "sens_positive"] = pos["sensitivity"]
        table.loc[lab, "spec_positive"] = pos["specificity"]
        table.loc[lab, "ppv_positive"] = pos["ppv"]
        table.loc[lab, "sens_not_negative"] = rule_out["sensitivity"]
        table.loc[lab, "npv_negative"] = rule_out["npv"]
        for k, v in status_counts(pt[:, i], c["t_low"], c["t_high"]).items():
            table.loc[lab, k] = v
        table.loc[lab, "ece_raw"] = ece(yt[:, i], pt_raw[:, i])
        table.loc[lab, "ece_calibrated"] = ece(yt[:, i], pt[:, i])
        status[:, i] = (pt[:, i] >= c["t_low"]).astype(int) + (pt[:, i] >= c["t_high"]).astype(int)
    table = table.round(4)
    table.to_csv(out / "metrics_test.csv")

    fidx = [labels.index(l) for l in focus]
    per_image = {
        "uncertain_per_image_all": float((status == 1).sum(1).mean()),
        "uncertain_per_image_focus": float((status[:, fidx] == 1).sum(1).mean()),
        "positive_per_image_all": float((status == 2).sum(1).mean()),
        "pct_images_any_uncertain_focus": float(((status[:, fidx] == 1).sum(1) > 0).mean()),
        "pct_images_all_focus_negative": float(((status[:, fidx] == 0).all(1)).mean()),
    }

    by_view = {}
    for view in sorted(test_df["view"].dropna().unique()):
        m = (test_df["view"] == view).values
        by_view[view] = {lab: safe_auc(yt[m, i], pt[m, i]) for i, lab in enumerate(labels)}
    by_view = pd.DataFrame(by_view).round(4)
    by_view.to_csv(out / "metrics_by_view.csv")

    preds = pd.DataFrame(pt, columns=[f"p_{l}" for l in labels])
    pd.concat([test_df[["Image Index", "Patient ID", "view"] + labels].reset_index(drop=True), preds],
              axis=1).to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "logits.npz", val_logits=lv, val_targets=yv, test_logits=lt, test_targets=yt,
                        labels=np.array(labels), test_ids=test_df["Image Index"].values,
                        test_patients=test_df["Patient ID"].values, test_views=test_df["view"].astype(str).values,
                        val_ids=np.asarray(val_ids if val_ids is not None else []))

    mean14 = float(table.loc[nih, "auroc"].mean())
    summary = {
        "test_mean_auroc_14": mean14,
        "test_mean_auroc_focus": float(table.loc[focus, "auroc"].mean()),
        "test_mean_auprc_14": float(table.loc[nih, "auprc"].mean()),
        "val_mean_auroc_14_calibrated": float(np.nanmean([calib["per_label"][l]["val_auroc"] for l in nih])),
        "test_mean_ece_calibrated": float(table["ece_calibrated"].mean()),
        "test_mean_ece_raw": float(table["ece_raw"].mean()),
        "test_mean_pct_uncertain_focus": float(table.loc[focus, "pct_uncertain"].mean()),
        "test_mean_auroc_14_AP": float(by_view["AP"].loc[nih].mean()) if "AP" in by_view else float("nan"),
        "test_mean_auroc_14_PA": float(by_view["PA"].loc[nih].mean()) if "PA" in by_view else float("nan"),
        **per_image, **(extra_metrics or {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    _figures(out, labels, focus, yt, pt, pt_raw, table, sweep)
    _report(out, table, by_view, sweep, summary, focus, meta, ecfg)

    if log_to_mlflow:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metrics({k: v for k, v in summary.items() if v == v})       # drop NaN
            mlflow.log_metrics({f"test_auroc_{l}": float(table.loc[l, "auroc"])
                                for l in labels if table.loc[l, "auroc"] == table.loc[l, "auroc"]})
            mlflow.log_metrics({f"test_pct_uncertain_{l}": float(table.loc[l, "pct_uncertain"]) for l in focus})
            mlflow.log_params({"eval.band_sensitivity": ecfg["band"]["sensitivity"],
                               "eval.band_specificity": ecfg["band"]["specificity"]}
                              if "eval.band.sensitivity" not in mlflow.active_run().data.params else {})
            mlflow.log_artifacts(str(out), artifact_path="eval")
    return summary


# ----------------------------------------------------------------------------- outputs
def _figures(out, labels, focus, yt, pt, pt_raw, table, sweep):
    idx = {l: labels.index(l) for l in labels}
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
        ax.set_title(lab, fontsize=9); ax.set_xlabel("predicted"); ax.set_ylabel("observed"); ax.legend(fontsize=7)
    fig.suptitle("Reliability (test)"); fig.tight_layout()
    fig.savefig(out / "reliability_focus.png", dpi=130); plt.close(fig)

    t = table.sort_values("auroc")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(t.index, t["auroc"], color=["#c0504d" if l in focus else "#4f81bd" for l in t.index])
    if "auroc_ci_low" in t:
        ax.errorbar(t["auroc"], range(len(t)), xerr=[t["auroc"] - t["auroc_ci_low"],
                    t["auroc_ci_high"] - t["auroc"]], fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUROC (test, 95% patient-bootstrap CI)")
    fig.tight_layout(); fig.savefig(out / "auroc_ci.png", dpi=130); plt.close(fig)

    cols = [c for c in focus if c in sweep.columns]
    if cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in cols:
            ax.plot(sweep.index, sweep[c] * 100, "o-", label=c)
        ax.set_xlabel("sensitivity = specificity target (validation)")
        ax.set_ylabel("% of images 'uncertain'"); ax.set_title("Uncertainty band width vs target")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "band_sweep.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    st = table.loc[focus, ["pct_negative", "pct_uncertain", "pct_positive"]] * 100
    st.plot.barh(stacked=True, ax=ax, color=["#9bbb59", "#f2c14e", "#c0504d"])
    ax.set_xlabel("% of test images"); ax.set_title("Status distribution (focus labels)")
    ax.legend(["negative", "uncertain", "positive"], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout(); fig.savefig(out / "status_focus.png", dpi=130); plt.close(fig)


def _report(out, table, by_view, sweep, summary, focus, meta, ecfg):
    cols = ["n_pos", "auroc"] + [c for c in ("auroc_ci_low", "auroc_ci_high") if c in table] + \
           ["auprc", "sens_positive", "spec_positive", "ppv_positive", "sens_not_negative", "npv_negative",
            "pct_negative", "pct_uncertain", "pct_positive", "ece_raw", "ece_calibrated"]
    b = ecfg["band"]
    lines = [f"# Evaluation report - {meta.get('experiment', meta.get('model_name', 'model'))}", "",
             f"Model: {meta.get('model_name')} | image size: {meta.get('image_size')} | "
             f"TTA: {meta.get('tta')} | best epoch: {meta.get('epoch', '-')}", "",
             f"**Test mean AUROC (14 NIH labels): {summary['test_mean_auroc_14']:.4f}**  ",
             f"Test mean AUROC (focus labels): {summary['test_mean_auroc_focus']:.4f}  ",
             f"Mean calibration error: {summary['test_mean_ece_raw']:.3f} raw -> "
             f"{summary['test_mean_ece_calibrated']:.3f} calibrated", "",
             "## Decision policy (fitted on validation only)", "",
             f"- p < t_low -> **negative**: t_low keeps {b['sensitivity']:.0%} of positives (rule-out).",
             f"- p >= t_high -> **positive**: {b['specificity']:.0%} specificity.",
             "- otherwise -> **uncertain** (the report must flag it).",
             f"- Per image: {summary['uncertain_per_image_focus']:.2f} uncertain focus labels on average; "
             f"{summary['pct_images_any_uncertain_focus']:.0%} of images have at least one.", "",
             "## Focus labels", "", table.loc[focus, cols].to_markdown(), "",
             "## All labels", "", table[cols].to_markdown(), "",
             "## Band width vs target (validation, % uncertain)", "", sweep.to_markdown(), "",
             "## AUROC by view (PA vs AP)", "", by_view.to_markdown(), "",
             "Column guide: sens/spec/ppv_positive = metrics of the 'positive' call; "
             "sens_not_negative = share of real cases NOT dismissed as negative; "
             "npv_negative = how often a 'negative' call is right."]
    (out / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
