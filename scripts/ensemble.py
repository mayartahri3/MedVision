"""Ensemble of already-evaluated experiments - no GPU needed.

Downloads each member's eval/logits.npz from MLflow (or reads local folders),
averages the logits, then runs the SAME evaluation as single models (Platt +
band fitted on validation, full test report). Logged as its own MLflow run.

Usage:
  python scripts/ensemble.py --experiments densenet121_320 convnext_tiny_320
  python scripts/ensemble.py --run-ids <id1> <id2>
  python scripts/ensemble.py --local /kaggle/working/densenet121_320/eval /kaggle/working/convnext_tiny_320/eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import load_config  # noqa: E402
from cxr.evaluation import evaluate_logits  # noqa: E402
from cxr.tracking import base_tags, log_config, setup_mlflow  # noqa: E402


def find_run(exp_name: str, mlflow_experiment: str) -> str:
    runs = mlflow.search_runs(experiment_names=[mlflow_experiment],
                              filter_string=f"tags.experiment = '{exp_name}' and metrics.test_mean_auroc_14 > 0",
                              order_by=["attributes.start_time DESC"])
    runs = runs[runs.get("tags.smoke_test", pd.Series(index=runs.index, dtype=object)).ne("true")]
    if runs.empty:
        raise SystemExit(f"No evaluated run found for experiment '{exp_name}'")
    return runs.iloc[0]["run_id"]


def load_member(run_id: str | None, local: str | None) -> tuple[str, dict]:
    if local:
        path = Path(local) / "logits.npz"
        name = Path(local).parent.name
    else:
        path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="eval/logits.npz"))
        name = mlflow.get_run(run_id).data.tags.get("experiment", run_id[:8])
    return name, dict(np.load(path, allow_pickle=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiments", nargs="+")
    g.add_argument("--run-ids", nargs="+")
    g.add_argument("--local", nargs="+")
    ap.add_argument("--name", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n_bootstrap is not None:
        cfg["eval"]["n_bootstrap"] = args.n_bootstrap
    if not args.no_mlflow or not args.local:
        setup_mlflow(cfg)

    if args.local:
        members = [load_member(None, d) for d in args.local]
        member_ids = ["local"] * len(members)
    else:
        member_ids = args.run_ids or [find_run(e, cfg["mlflow"]["experiment"]) for e in args.experiments]
        members = [load_member(r, None) for r in member_ids]
    names = [m[0] for m in members]
    print("Members:", dict(zip(names, member_ids)))

    ref = members[0][1]
    for name, d in members[1:]:
        if not (np.array_equal(d["test_ids"], ref["test_ids"]) and np.array_equal(d["labels"], ref["labels"])):
            raise SystemExit(f"{name}: test images or labels differ from {names[0]} - cannot ensemble")
        if len(d["val_ids"]) and len(ref["val_ids"]) and not np.array_equal(d["val_ids"], ref["val_ids"]):
            raise SystemExit(f"{name}: validation images differ from {names[0]}")
    lv = np.mean([d["val_logits"] for _, d in members], axis=0)
    lt = np.mean([d["test_logits"] for _, d in members], axis=0)
    labels = [str(x) for x in ref["labels"]]
    test_df = pd.DataFrame({"Image Index": ref["test_ids"], "Patient ID": ref["test_patients"],
                            "view": ref["test_views"]})
    for i, lab in enumerate(labels):
        test_df[lab] = ref["test_targets"][:, i]

    ens_name = args.name or "ensemble-" + "+".join(names)
    work = Path(cfg["eval"]["out_dir"]).parent
    out = work / ens_name / "eval"
    meta = {"experiment": ens_name, "model_name": "ensemble(" + ", ".join(names) + ")",
            "image_size": "mixed", "tta": "as members", "members": names, "member_run_ids": member_ids}

    run = None
    if not args.no_mlflow:
        run = mlflow.start_run(run_name=ens_name)
        tags = base_tags(cfg)
        tags.update({"experiment": ens_name, "model": "ensemble", "image_size": "mixed", "type": "ensemble",
                     "members": ",".join(names), "member_run_ids": ",".join(member_ids),
                     "mlflow.note.content": f"Mean of logits of: {', '.join(names)}"})
        mlflow.set_tags(tags)
        log_config(cfg)
        mlflow.log_params({"ensemble.n_members": len(names)})
    try:
        summary = evaluate_logits(lv, ref["val_targets"], lt, ref["test_targets"], test_df, labels, cfg, out,
                                  meta, val_ids=ref["val_ids"], log_to_mlflow=run is not None)
    finally:
        if run is not None:
            mlflow.end_run()
    print(f"{ens_name}: test mean AUROC (14) = {summary['test_mean_auroc_14']:.4f} -> {out}")


if __name__ == "__main__":
    main()
