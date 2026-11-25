"""MLflow tracing: one run per experiment, shared by train / evaluate / Grad-CAM.

- MLFLOW_TRACKING_URI set (e.g. DagsHub) -> log there.
- Otherwise -> local SQLite DB + artifact folder next to the outputs. Browse with
      mlflow ui --backend-store-uri sqlite:///<work>/mlflow.db

train.py creates the run and writes its id to <checkpoint_dir>/mlflow_run_id.txt;
evaluate.py and gradcam_eval.py reopen the SAME run, so each model's training
curves, test evaluation, calibration and explainability live together.
"""
from __future__ import annotations

import inspect
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import yaml

from cxr.config import PROJECT_ROOT


def setup_mlflow(cfg: dict[str, Any]) -> str:
    name = cfg["mlflow"]["experiment"]
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(name)
        return os.environ["MLFLOW_TRACKING_URI"]

    work = Path(cfg["train"]["checkpoint_dir"]).parent
    if cfg.get("experiment", {}).get("name", "default") != "default":
        work = work.parent            # one DB for all experiments
    work.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(work / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(work / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)
    return uri


def run_id_file(cfg: dict[str, Any]) -> Path:
    return Path(cfg["train"]["checkpoint_dir"]) / "mlflow_run_id.txt"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def base_tags(cfg: dict[str, Any]) -> dict[str, str]:
    import torch
    exp = cfg.get("experiment", {})
    return {
        "experiment": exp.get("name", "default"),
        "mlflow.note.content": exp.get("description", ""),
        "note": exp.get("note", ""),
        "model": cfg["model"]["name"],
        "image_size": str(cfg["preprocess"]["image_size"]),
        "dataset": "NIH ChestX-ray14",
        "split": "official test list + 10% patient-level validation",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "platform": "kaggle" if Path("/kaggle").exists() else platform.system().lower(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v))[:500]
        else:
            out[key] = v
    return out


def start_run(cfg: dict[str, Any], stage: str, run_id: str | None = None, new: bool = False):
    """Open the experiment's run: create it (train) or resume it (evaluate / explain)."""
    setup_mlflow(cfg)
    rid_file = run_id_file(cfg)
    if not new:
        run_id = run_id or (rid_file.read_text().strip() if rid_file.exists() else None)
    kwargs: dict[str, Any] = {}
    if cfg["mlflow"].get("system_metrics") and \
            "log_system_metrics" in inspect.signature(mlflow.start_run).parameters:
        kwargs["log_system_metrics"] = True
    if run_id:
        run = mlflow.start_run(run_id=run_id, **kwargs)
    else:
        run = mlflow.start_run(run_name=cfg.get("experiment", {}).get("name", "default"), **kwargs)
        rid_file.parent.mkdir(parents=True, exist_ok=True)
        rid_file.write_text(run.info.run_id)
        mlflow.set_tags(base_tags(cfg))
    mlflow.set_tag(f"stage.{stage}", "done")
    return run


def log_config(cfg: dict[str, Any]) -> None:
    """All hyper-parameters as params + the exact resolved YAML as an artifact."""
    skip = ("paths.", "data.", "labels.")
    params = {k: v for k, v in flatten(cfg).items() if not k.startswith(skip)}
    items = list(params.items())
    for i in range(0, len(items), 90):          # MLflow batch limit is 100 params
        mlflow.log_params(dict(items[i:i + 90]))
    mlflow.log_text(yaml.safe_dump(cfg, sort_keys=False), "config/resolved_config.yaml")


def log_dataset(df, name: str, context: str) -> None:
    """Record which data the run used (schema + digest), visible in the MLflow 'Datasets' panel."""
    try:
        ds = mlflow.data.from_pandas(df, name=name)
        mlflow.log_input(ds, context=context)
    except Exception as e:  # older servers / clients
        print(f"(dataset logging skipped: {e})")


def log_torch_model(model, cfg: dict[str, Any]) -> None:
    if not cfg["mlflow"].get("log_model"):
        return
    import mlflow.pytorch
    reg = f"cxr-{cfg.get('experiment', {}).get('name', 'default')}" \
        if cfg["mlflow"].get("register_model") else None
    try:
        import numpy as np
        params = inspect.signature(mlflow.pytorch.log_model).parameters
        key = "name" if "name" in params else "artifact_path"   # MLflow 3 vs 2
        size = cfg["preprocess"]["image_size"]
        ch = 1 if cfg["preprocess"]["normalization"] == "xrv" else cfg["preprocess"]["in_channels"]
        kwargs = {key: "pytorch_model", "registered_model_name": reg,
                  # example input -> MLflow stores the input/output signature of the model
                  "input_example": np.zeros((1, ch, size, size), dtype=np.float32)}
        if "serialization_format" in params:
            kwargs["serialization_format"] = "pickle"   # robust for any torchvision model
        mlflow.pytorch.log_model(model.cpu().eval(), **kwargs)
        print("Model logged to MLflow" + (f" and registered as {reg}" if reg else ""))
    except Exception as e:
        print(f"(model flavour logging skipped: {e}) - checkpoint is still logged as an artifact")
