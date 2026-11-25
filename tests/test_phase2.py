import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cxr.explain.gradcam import GradCAM, forward_with_features, overlay  # noqa: E402
from cxr.metrics import (apply_platt, bootstrap_auroc_ci, ece, fit_platt,  # noqa: E402
                         operating_points, safe_auc)
from cxr.models.factory import build_model, load_checkpoint, save_checkpoint  # noqa: E402
from gradcam_eval import in_box, to_model_coords  # noqa: E402

RNG = np.random.default_rng(0)


def _toy(n=4000, prev=0.05, shift=2.0):
    y = (RNG.random(n) < prev).astype(int)
    z = RNG.normal(0, 1, n) + shift * y + 1.5  # +1.5 mimics the pos_weight upward bias
    return y, z


def test_platt_fixes_biased_logits():
    y, z = _toy()
    raw = 1 / (1 + np.exp(-z))
    a, b = fit_platt(z, y)
    cal = apply_platt(z, a, b)
    assert ece(y, cal) < ece(y, raw) / 3
    assert abs(cal.mean() - y.mean()) < 0.01


def test_operating_points_ordered_and_meaningful():
    y, z = _toy()
    p = apply_platt(z, *fit_platt(z, y))
    ops = operating_points(y, p, 0.95, 0.95)
    assert ops["t_low"] <= ops["t_high"]
    assert (p[y == 1] >= ops["t_low"]).mean() >= 0.95       # <=5% of positives called negative
    assert (p[y == 0] >= ops["t_high"]).mean() <= 0.05 + 1e-9


def test_bootstrap_ci_contains_point_estimate():
    y, z = _toy(n=2000, prev=0.2)
    patients = np.repeat(np.arange(500), 4)
    lo, hi = bootstrap_auroc_ci(y, z, patients, n_boot=100)
    assert lo < safe_auc(y, z) < hi


def test_safe_auc_single_class_is_nan():
    assert np.isnan(safe_auc(np.zeros(10), np.random.rand(10)))


def test_forward_with_features_matches_model():
    for name in ("densenet121", "resnet50", "convnext_tiny", "efficientnet_v2_s", "efficientnet_b0"):
        m = build_model(name, 15, pretrained=False).eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            ref = m(x)
            logits, A = forward_with_features(m, name, x)
        assert torch.allclose(ref, logits, atol=1e-4), name
        assert A.shape[-1] == 7, name


def test_gradcam_shapes_and_range():
    m = build_model("densenet121", 15, pretrained=False).eval()
    logits, cams = GradCAM(m, "densenet121")(torch.randn(1, 3, 224, 224), [1, 6])
    assert logits.shape == (15,) and cams.shape == (2, 224, 224)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert overlay(np.zeros((320, 320), np.uint8), cams[0]).shape == (320, 320, 3)


def test_checkpoint_roundtrip(tmp_path):
    cfg = {"model": {"name": "densenet121", "dropout": 0.2},
           "preprocess": {"image_size": 224, "cache_size": 320, "normalization": "imagenet",
                          "in_channels": 3}}
    m = build_model("densenet121", 15, pretrained=False)
    save_checkpoint(tmp_path / "b.pt", m, cfg, [f"l{i}" for i in range(15)], 3, {"val_mean_auroc": 0.8})
    m2, ck = load_checkpoint(tmp_path / "b.pt")
    assert ck["epoch"] == 3 and len(ck["labels"]) == 15
    x = torch.randn(1, 3, 224, 224)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-5)


def test_bbox_mapping_square_and_padded():
    bx, by, bw, bh, k = to_model_coords((512, 512, 100, 100), (1024, 1024), 224)
    assert np.isclose(bx, 112) and np.isclose(bw, 21.875)
    bx, by, *_ = to_model_coords((0, 0, 10, 10), (800, 1000), 200)   # padded top/bottom by 100
    assert np.isclose(by, 20) and np.isclose(bx, 0)
    assert in_box(5, 5, 0, 0, 10, 10, 0) and not in_box(20, 5, 0, 0, 10, 10, 2)


# ---------------------------------------------------------------- experiments / evaluation core
def test_band_collapses_for_strong_classifier():
    from cxr.metrics import band_thresholds
    y = np.r_[np.zeros(500), np.ones(500)]
    p = np.r_[RNG.uniform(0, 0.4, 500), RNG.uniform(0.6, 1, 500)]   # perfectly separable
    b = band_thresholds(y, p, 0.9, 0.9)
    assert b["t_low"] == b["t_high"]


def test_every_experiment_config_builds():
    from cxr.config import list_experiments, load_config
    for name in list_experiments():
        cfg = load_config(experiment=name)
        assert cfg["experiment"]["name"] == name
        assert f"/{name}/" in cfg["train"]["checkpoint_dir"]
        build_model(cfg["model"]["name"], 15, pretrained=False)


def test_evaluate_logits_end_to_end(tmp_path):
    import pandas as pd
    from cxr.config import all_labels, load_config
    from cxr.evaluation import evaluate_logits
    cfg = load_config()
    cfg["eval"]["n_bootstrap"] = 5
    labels = all_labels(cfg)
    n = 600
    yv = (RNG.random((n, len(labels))) < 0.2).astype(float)
    yt = (RNG.random((n, len(labels))) < 0.2).astype(float)
    lv = RNG.normal(0, 1, yv.shape) + 1.5 * yv + 1.0
    lt = RNG.normal(0, 1, yt.shape) + 1.5 * yt + 1.0
    df = pd.DataFrame({"Image Index": [f"{i}.png" for i in range(n)], "Patient ID": np.arange(n) // 3,
                       "view": np.where(np.arange(n) % 2, "PA", "AP")})
    for i, l in enumerate(labels):
        df[l] = yt[:, i]
    s = evaluate_logits(lv, yv, lt, yt, df, labels, cfg, tmp_path, {"model_name": "toy"}, log_to_mlflow=False)
    assert 0.75 < s["test_mean_auroc_14"] < 0.9
    assert s["test_mean_ece_calibrated"] < s["test_mean_ece_raw"]
    for f in ("calibration.json", "logits.npz", "eval_report.md", "band_sweep_val.csv", "status_focus.png"):
        assert (tmp_path / f).exists(), f


def test_tta_keeps_shape():
    from cxr.evaluation import tta_logits
    m = build_model("densenet121", 15, pretrained=False).eval()
    with torch.no_grad():
        assert tta_logits(m, torch.randn(2, 3, 224, 224)).shape == (2, 15)
