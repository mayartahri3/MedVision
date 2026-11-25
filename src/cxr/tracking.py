"""MLflow setup shared by train / evaluate.

- MLFLOW_TRACKING_URI set (e.g. DagsHub) -> log there.
- Otherwise -> local SQLite DB + artifact folder next to the checkpoints, so the
  whole MLflow history is saved in the Kaggle notebook output. Open it later with:
      mlflow ui --backend-store-uri sqlite:///<work>/mlflow.db
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow


def setup_mlflow(cfg: dict[str, Any]) -> str:
    name = cfg["train"]["mlflow_experiment"]
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_experiment(name)
        return os.environ["MLFLOW_TRACKING_URI"]

    work = Path(cfg["train"]["checkpoint_dir"]).parent
    work.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(work / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(work / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)
    return uri
