"""Step 5 - evaluate one experiment and log it to the SAME MLflow run as its training.

- predicts VAL and TEST (with and without test-time augmentation)
- Platt calibration + uncertainty band fitted on VAL (see cxr/evaluation.py)
- full TEST report: AUROC with patient-bootstrap CIs, decision metrics, status
  distribution, per-image uncertainty, calibration, PA vs AP, band-width sweep
- logits saved (eval/logits.npz) so ensembles can be built later without a GPU

Usage: python scripts/evaluate.py --experiment densenet121_320
       python scripts/evaluate.py --checkpoint path/best.pt --run-id <id>   # attach to any run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxr.config import load_config  # noqa: E402
from cxr.data.dataset import make_loaders  # noqa: E402
from cxr.evaluation import evaluate_logits, tta_logits  # noqa: E402
from cxr.metrics import mean_auroc, sigmoid  # noqa: E402
from cxr.models.factory import load_checkpoint  # noqa: E402
from cxr.tracking import start_run  # noqa: E402
from train import predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--run-id", default=None, help="MLflow run to log into (default: the experiment's run)")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=None)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.experiment)
    if args.batch_size:
        cfg["loader"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["loader"]["num_workers"] = args.workers
    if args.n_bootstrap is not None:
        cfg["eval"]["n_bootstrap"] = args.n_bootstrap
    use_tta = cfg["eval"]["tta"] and not args.no_tta
    out = Path(cfg["eval"]["out_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or Path(cfg["train"]["checkpoint_dir"]) / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device)
    labels = ckpt["labels"]
    nih = [l for l in cfg["labels"]["nih"] if l in labels]
    # evaluate with the preprocessing the model was trained with
    cfg["preprocess"].update(ckpt["preprocess"])
    print(f"{ckpt.get('experiment', '?')}: {ckpt['model_name']} @ {cfg['preprocess']['image_size']} px, "
          f"epoch {ckpt['epoch']}, val mean AUROC {ckpt['metrics']['val_mean_auroc']:.4f}")

    loaders = make_loaders(cfg, subset=args.subset)
    amp = device.type == "cuda"
    lv, yv = predict(model, loaders["val"], device, amp)
    lt, yt = predict(model, loaders["test"], device, amp)
    extra = {"test_mean_auroc_14_no_tta": mean_auroc(yt, sigmoid(lt), labels, nih)}
    if use_tta:
        lv, _ = predict(model, loaders["val"], device, amp, tta_fn=tta_logits)
        lt, _ = predict(model, loaders["test"], device, amp, tta_fn=tta_logits)
        extra["test_mean_auroc_14_tta"] = mean_auroc(yt, sigmoid(lt), labels, nih)
        print(f"TTA: {extra['test_mean_auroc_14_no_tta']:.4f} -> {extra['test_mean_auroc_14_tta']:.4f}")

    meta = {"experiment": ckpt.get("experiment", cfg["experiment"]["name"]), "model_name": ckpt["model_name"],
            "image_size": cfg["preprocess"]["image_size"], "cache_size": cfg["preprocess"]["cache_size"],
            "normalization": cfg["preprocess"]["normalization"], "tta": use_tta, "epoch": ckpt["epoch"],
            "checkpoint": str(ckpt_path)}

    run = None if args.no_mlflow else start_run(cfg, "evaluate", run_id=args.run_id)
    try:
        summary = evaluate_logits(lv, yv, lt, yt, loaders["test"].dataset.df, labels, cfg, out, meta,
                                  val_ids=loaders["val"].dataset.df["Image Index"].values,
                                  extra_metrics=extra, log_to_mlflow=run is not None)
    finally:
        if run is not None:
            mlflow.end_run()
    for k in ("test_mean_auroc_14", "test_mean_auroc_focus", "test_mean_ece_calibrated",
              "uncertain_per_image_focus", "pct_images_any_uncertain_focus"):
        print(f"{k:34s} {summary[k]:.4f}")
    print(f"Outputs -> {out}")


if __name__ == "__main__":
    main()
