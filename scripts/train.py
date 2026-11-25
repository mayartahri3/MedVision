"""Step 4 - train one experiment and trace everything in MLflow.

Usage (Kaggle GPU):  python scripts/train.py --experiment densenet121_320
Smoke test:          python scripts/train.py --experiment densenet121_320 --epochs 1 --max-steps 20 --subset 640
List experiments:    python scripts/train.py --list

Logged to the experiment's MLflow run: resolved config + all hyper-parameters,
git commit, dataset digest, GPU/CPU usage, per-epoch loss / LR / val AUROC per
label, training history, best checkpoint and the PyTorch model.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import all_labels, list_experiments, load_config  # noqa: E402
from cxr.data.dataset import make_loaders  # noqa: E402
from cxr.metrics import mean_auroc, per_label_auroc  # noqa: E402
from cxr.models.factory import build_model, count_parameters, save_checkpoint  # noqa: E402
from cxr.tracking import log_config, log_dataset, log_torch_model, start_run  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, amp: bool, max_steps: int | None = None, tta_fn=None):
    model.eval()
    logits, targets = [], []
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = tta_fn(model, x) if tta_fn else model(x)
        logits.append(out.float().cpu()); targets.append(batch["target"])
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


class EMA:
    """Exponential moving average of weights (and BN buffers)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def make_scheduler(optimizer, tcfg, steps_per_epoch: int):
    if tcfg["scheduler"] == "cosine":
        warm = tcfg["warmup_epochs"] * steps_per_epoch
        total = max(tcfg["epochs"] * steps_per_epoch, warm + 1)

        def f(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            t = (step - warm) / max(total - warm, 1)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f), "step"
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=tcfg["lr_patience"]), "epoch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--experiment", default=None, help="name from configs/experiments.yaml")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="cap batches per epoch (smoke test)")
    ap.add_argument("--subset", type=int, default=None, help="images per split (smoke test)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tag the run as a smoke test")
    args = ap.parse_args()

    if args.list:
        for k, v in list_experiments().items():
            print(f"{k:24s} {v.get('description', '')}")
        return

    cfg = load_config(args.config, args.experiment)
    tcfg = cfg["train"]
    if args.epochs:
        tcfg["epochs"] = args.epochs
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    if args.batch_size:
        cfg["loader"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["loader"]["num_workers"] = args.workers
    if args.smoke:
        cfg["experiment"]["name"] += "-smoke"
        for sec, key in (("train", "checkpoint_dir"), ("eval", "out_dir"), ("explain", "out_dir")):
            cfg[sec][key] = cfg[sec][key].replace(f"/{args.experiment}/", f"/{args.experiment}-smoke/") \
                if args.experiment else cfg[sec][key] + "-smoke"
    set_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = all_labels(cfg)
    nih = cfg["labels"]["nih"]
    print(f"Experiment: {cfg['experiment']['name']} | device: {device} | model: {cfg['model']['name']} "
          f"| {cfg['preprocess']['image_size']} px | {len(labels)} labels")

    loaders = make_loaders(cfg, subset=args.subset)
    print({k: len(v.dataset) for k, v in loaders.items()})

    model = build_model(cfg["model"]["name"], len(labels), cfg["model"]["pretrained"],
                        cfg["model"]["dropout"]).to(device)
    ema = EMA(model, tcfg["ema_decay"]) if tcfg["ema_decay"] > 0 else None

    pw_file = Path(cfg["paths"]["processed_dir"]) / "pos_weight.csv"
    pw = pd.read_csv(pw_file, index_col=0)["pos_weight"].reindex(labels).fillna(1.0)
    pos_weight = torch.tensor(pw.clip(upper=tcfg["pos_weight_clip"]).values,
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ls = tcfg["label_smoothing"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps_per_epoch = min(len(loaders["train"]), args.max_steps or 10**9)
    scheduler, sched_mode = make_scheduler(optimizer, tcfg, steps_per_epoch)
    amp = tcfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    run = start_run(cfg, "train", new=True)
    print(f"MLflow run: {run.info.run_id} ({mlflow.get_tracking_uri()})")
    if args.smoke:
        mlflow.set_tag("smoke_test", "true")
    log_config(cfg)
    mlflow.log_params({"n_params_m": round(count_parameters(model) / 1e6, 2),
                       "n_train": len(loaders["train"].dataset), "n_val": len(loaders["val"].dataset),
                       "device": str(device), "amp_effective": amp})
    splits = loaders["train"].dataset.df
    log_dataset(splits[["Image Index", "Patient ID", "view"] + labels], "nih-train", "training")
    log_dataset(loaders["val"].dataset.df[["Image Index", "Patient ID", "view"] + labels],
                "nih-val", "validation")

    best, bad_epochs, history, global_step = -1.0, 0, [], 0
    t_start = time.time()
    failed = False
    try:
        for epoch in range(1, tcfg["epochs"] + 1):
            model.train()
            t0, running, n = time.time(), 0.0, 0
            for step, batch in enumerate(loaders["train"]):
                if args.max_steps and step >= args.max_steps:
                    break
                x = batch["image"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                if ls > 0:
                    y = y * (1 - ls) + 0.5 * ls
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema:
                    ema.update(model)
                if sched_mode == "step":
                    scheduler.step()
                running += loss.item() * len(x); n += len(x); global_step += 1
                if step % 200 == 0:
                    print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f}", flush=True)
                    mlflow.log_metric("train_loss_step", loss.item(), step=global_step)

            eval_model = ema.model if ema else model
            logits, y_val = predict(eval_model, loaders["val"], device, amp, args.max_steps)
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(
                torch.tensor(logits), torch.tensor(y_val), pos_weight=pos_weight.cpu()))
            probs = 1 / (1 + np.exp(-logits))
            val_auc = mean_auroc(y_val, probs, labels, nih)
            aucs = per_label_auroc(y_val, probs, labels)
            if sched_mode == "epoch":
                scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)
            lr = optimizer.param_groups[0]["lr"]
            train_loss = running / max(n, 1)
            dt = time.time() - t0
            print(f"Epoch {epoch:2d} | train {train_loss:.4f} | val loss {val_loss:.4f} | "
                  f"val mean AUROC (14) {val_auc:.4f} | lr {lr:.1e} | {dt / 60:.1f} min", flush=True)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_mean_auroc": val_auc,
                                "lr": lr, "epoch_minutes": dt / 60}, step=epoch)
            mlflow.log_metrics({f"val_auroc_{k}": v for k, v in aucs.items() if not np.isnan(v)}, step=epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "val_mean_auroc": val_auc, "lr": lr, **aucs})

            if val_auc > best:
                best, bad_epochs = val_auc, 0
                save_checkpoint(best_path, eval_model, cfg, labels, epoch, {"val_mean_auroc": val_auc, **aucs})
                mlflow.log_metric("best_epoch", epoch)
                print(f"  -> new best, saved {best_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= tcfg["patience"]:
                    print(f"Early stopping: no improvement for {bad_epochs} epochs")
                    break
    except BaseException:
        failed = True
        raise
    finally:
        hist = pd.DataFrame(history)
        hist.to_csv(ckpt_dir / "history.csv", index=False)
        if len(hist):
            _plot_history(hist, ckpt_dir / "training_curves.png")
            mlflow.log_artifact(str(ckpt_dir / "training_curves.png"), "training")
        mlflow.log_artifact(str(ckpt_dir / "history.csv"), "training")
        mlflow.log_metrics({"best_val_mean_auroc": best, "train_minutes": (time.time() - t_start) / 60})
        (ckpt_dir / "best_summary.json").write_text(json.dumps(
            {"best_val_mean_auroc": best, "run_id": run.info.run_id}, indent=2))
        if failed:
            mlflow.end_run(status="FAILED")

    if best_path.exists():
        mlflow.log_artifact(str(best_path), "checkpoint")
        from cxr.models.factory import load_checkpoint
        log_torch_model(load_checkpoint(best_path, "cpu")[0], cfg)
    mlflow.end_run()
    print(f"Best val mean AUROC: {best:.4f}  -> {best_path}")


def _plot_history(h: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train"); ax[0].plot(h.epoch, h.val_loss, "s-", label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_mean_auroc, "o-", color="#c0504d"); ax[1].set_title("Val mean AUROC (14)")
    ax[2].plot(h.epoch, h.lr, "o-", color="#9bbb59"); ax[2].set_yscale("log"); ax[2].set_title("Learning rate")
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
