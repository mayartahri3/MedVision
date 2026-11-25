"""Step 4 - train the multi-label chest X-ray classifier.

Usage (Kaggle GPU):  python scripts/train.py
Smoke test:          python scripts/train.py --epochs 1 --max-steps 20 --subset 512

MLflow: set MLFLOW_TRACKING_URI (+ MLFLOW_TRACKING_USERNAME / _PASSWORD for
DagsHub). Without it, runs go to <work>/mlflow.db (kept in the notebook output).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.tracking import setup_mlflow  # noqa: E402
from cxr.config import all_labels, load_config  # noqa: E402
from cxr.data.dataset import make_loaders  # noqa: E402
from cxr.metrics import mean_auroc, per_label_auroc  # noqa: E402
from cxr.models.factory import build_model, save_checkpoint  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, amp: bool, max_steps: int | None = None):
    model.eval()
    logits, targets = [], []
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(batch["image"].to(device, non_blocking=True))
        logits.append(out.float().cpu()); targets.append(batch["target"])
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="cap batches per epoch (smoke test)")
    ap.add_argument("--subset", type=int, default=None, help="images per split (smoke test)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="override loader.batch_size")
    ap.add_argument("--workers", type=int, default=None, help="override loader.num_workers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["loader"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["loader"]["num_workers"] = args.workers
    tcfg = cfg["train"]
    if args.epochs:
        tcfg["epochs"] = args.epochs
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    set_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = all_labels(cfg)
    nih = cfg["labels"]["nih"]
    print(f"Device: {device} | labels: {len(labels)} | model: {cfg['model']['name']}")

    loaders = make_loaders(cfg, subset=args.subset)
    print({k: len(v.dataset) for k, v in loaders.items()})

    model = build_model(cfg["model"]["name"], len(labels), cfg["model"]["pretrained"],
                        cfg["model"]["dropout"]).to(device)

    pw_file = Path(cfg["paths"]["processed_dir"]) / "pos_weight.csv"
    pw = pd.read_csv(pw_file, index_col=0)["pos_weight"].reindex(labels).fillna(1.0)
    pos_weight = torch.tensor(pw.clip(upper=tcfg["pos_weight_clip"]).values,
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=tcfg["lr_patience"])
    amp = tcfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    print("MLflow tracking:", setup_mlflow(cfg))

    best, bad_epochs, history = -1.0, 0, []
    with mlflow.start_run(run_name=args.run_name or f"{cfg['model']['name']}-{cfg['preprocess']['image_size']}"):
        mlflow.log_params({
            "model": cfg["model"]["name"], "pretrained": cfg["model"]["pretrained"],
            "image_size": cfg["preprocess"]["image_size"], "batch_size": cfg["loader"]["batch_size"],
            "lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"], "dropout": cfg["model"]["dropout"],
            "pos_weight_clip": tcfg["pos_weight_clip"], "epochs_max": tcfg["epochs"], "amp": amp,
            "n_train": len(loaders["train"].dataset), "n_val": len(loaders["val"].dataset),
        })

        for epoch in range(1, tcfg["epochs"] + 1):
            model.train()
            t0, running, n = time.time(), 0.0, 0
            for step, batch in enumerate(loaders["train"]):
                if args.max_steps and step >= args.max_steps:
                    break
                x = batch["image"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item() * len(x); n += len(x)
                if step % 200 == 0:
                    print(f"  epoch {epoch} step {step}/{len(loaders['train'])} loss {loss.item():.4f}", flush=True)

            logits, y_val = predict(model, loaders["val"], device, amp, args.max_steps)
            probs = 1 / (1 + np.exp(-logits))
            val_auc = mean_auroc(y_val, probs, labels, nih)
            aucs = per_label_auroc(y_val, probs, labels)
            scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)
            lr = optimizer.param_groups[0]["lr"]
            train_loss = running / max(n, 1)
            dt = time.time() - t0
            print(f"Epoch {epoch:2d} | loss {train_loss:.4f} | val mean AUROC (14) {val_auc:.4f} "
                  f"| lr {lr:.1e} | {dt / 60:.1f} min", flush=True)

            mlflow.log_metrics({"train_loss": train_loss, "val_mean_auroc": val_auc, "lr": lr,
                                "epoch_minutes": dt / 60}, step=epoch)
            mlflow.log_metrics({f"val_auroc_{k}": v for k, v in aucs.items() if not np.isnan(v)}, step=epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_mean_auroc": val_auc, **aucs})

            if val_auc > best:
                best, bad_epochs = val_auc, 0
                save_checkpoint(best_path, model, cfg, labels, epoch, {"val_mean_auroc": val_auc, **aucs})
                print(f"  -> new best, saved {best_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= tcfg["patience"]:
                    print(f"Early stopping: no improvement for {bad_epochs} epochs")
                    break

        pd.DataFrame(history).to_csv(ckpt_dir / "history.csv", index=False)
        mlflow.log_metric("best_val_mean_auroc", best)
        mlflow.log_artifact(str(ckpt_dir / "history.csv"))
        if best_path.exists():
            mlflow.log_artifact(str(best_path), artifact_path="model")
        (ckpt_dir / "best_summary.json").write_text(json.dumps({"best_val_mean_auroc": best}, indent=2))
        print(f"Best val mean AUROC: {best:.4f}  -> {best_path}")


if __name__ == "__main__":
    main()
