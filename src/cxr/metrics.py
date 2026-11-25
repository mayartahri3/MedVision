"""Multi-label metrics, calibration and patient-level bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def per_label_auroc(y: np.ndarray, p: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {lab: safe_auc(y[:, i], p[:, i]) for i, lab in enumerate(labels)}


def mean_auroc(y: np.ndarray, p: np.ndarray, labels: list[str], subset: list[str]) -> float:
    aucs = per_label_auroc(y, p, labels)
    vals = [aucs[l] for l in subset if not np.isnan(aucs[l])]
    return float(np.mean(vals)) if vals else float("nan")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid=np.linspace(0.5, 5.0, 91)) -> float:
    """Single-label temperature by grid search on NLL (robust, no optimiser state)."""
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a * z + b), fitted on validation logits.

    Preferred over temperature scaling here: pos_weight in the loss shifts every
    logit upward (~log(pos_weight)), which a pure scale cannot undo.
    """
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logits.reshape(-1, 1), y.astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if a <= 0:  # model anti-correlated on val: never let calibration reverse the ranking
        print(f"WARNING: Platt slope {a:.3f} <= 0; falling back to identity calibration")
        return 1.0, 0.0
    return a, b


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid(a * logits + b)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error for one binary label."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def band_thresholds(y: np.ndarray, p: np.ndarray, sensitivity: float,
                    specificity: float) -> dict[str, float]:
    """Uncertainty band [t_low, t_high] fitted on validation.

    t_low : highest threshold that still keeps `sensitivity` of positives >= t_low
            -> p < t_low is called NEGATIVE (rule-out)
    t_high: lowest threshold whose false-positive rate is <= 1 - specificity
            -> p >= t_high is called POSITIVE
    In between -> UNCERTAIN. If the model is good enough that both targets can be
    met by one threshold (t_low > t_high), there is no uncertain zone.
    """
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0, 1)
    t_low = float(thr[np.argmax(tpr >= sensitivity)])
    ok = np.where(fpr <= 1 - specificity)[0]
    t_high = float(thr[ok[-1]]) if len(ok) else 1.0
    if t_low > t_high:
        t_low = t_high = (t_low + t_high) / 2
    return {"t_low": t_low, "t_high": t_high}


def operating_points(y: np.ndarray, p: np.ndarray, sens_target: float,
                     spec_target: float) -> dict[str, float]:
    """Band + `threshold` (= t_high, the 'positive' cut-off) for backward compatibility."""
    b = band_thresholds(y, p, sens_target, spec_target)
    return {"threshold": b["t_high"], **b}


def status_counts(p: np.ndarray, t_low: float, t_high: float) -> dict[str, float]:
    return {"pct_negative": float(np.mean(p < t_low)),
            "pct_uncertain": float(np.mean((p >= t_low) & (p < t_high))),
            "pct_positive": float(np.mean(p >= t_high))}


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = p >= t
    tp = np.sum(pred & (y == 1)); tn = np.sum(~pred & (y == 0))
    fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1), "f1": _f1(y, pred)}


def bootstrap_auroc_ci(y: np.ndarray, p: np.ndarray, patients: np.ndarray,
                       n_boot: int, seed: int = 0) -> tuple[float, float]:
    """95% CI resampling PATIENTS (images of one patient are correlated)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patients, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[k] for k in pick])
        a = safe_auc(y[idx], p[idx])
        if not np.isnan(a):
            stats.append(a)
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summary_table(y, p, labels, patients=None, n_boot=0) -> pd.DataFrame:
    rows = []
    for i, lab in enumerate(labels):
        r = {"label": lab, "n_pos": int(y[:, i].sum()), "auroc": safe_auc(y[:, i], p[:, i]),
             "auprc": float(average_precision_score(y[:, i], p[:, i])) if y[:, i].sum() else float("nan")}
        if n_boot and patients is not None:
            r["auroc_ci_low"], r["auroc_ci_high"] = bootstrap_auroc_ci(y[:, i], p[:, i], patients, n_boot)
        rows.append(r)
    return pd.DataFrame(rows).set_index("label")
