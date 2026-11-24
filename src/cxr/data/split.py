"""Patient-level splitting.

The same patient can have dozens of follow-up X-rays in NIH. If they end up
in both train and test, the model recognises the *patient* rather than the
*disease* and the metrics are inflated. All splits here are by Patient ID.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_splits(df: pd.DataFrame, test_images: set[str] | None,
                  val_fraction: float, seed: int,
                  patient_col: str = "Patient ID",
                  image_col: str = "Image Index") -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)

    if test_images:  # official NIH test list (patient-disjoint by construction)
        is_test = out[image_col].isin(test_images)
    else:            # fallback: random 20% of patients
        pats = out[patient_col].unique()
        test_pats = set(rng.choice(pats, size=int(0.2 * len(pats)), replace=False))
        is_test = out[patient_col].isin(test_pats)

    out["split"] = np.where(is_test, "test", "train")

    trainval_pats = out.loc[~is_test, patient_col].unique()
    rng.shuffle(trainval_pats)
    n_val = int(round(val_fraction * len(trainval_pats)))
    val_pats = set(trainval_pats[:n_val])
    out.loc[(~is_test) & out[patient_col].isin(val_pats), "split"] = "val"

    check_no_leakage(out, patient_col)
    return out


def check_no_leakage(df: pd.DataFrame, patient_col: str = "Patient ID") -> None:
    splits = df.groupby(patient_col)["split"].nunique()
    leaked = splits[splits > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} patients appear in more than one split, "
                             f"e.g. {leaked.index[:5].tolist()}")
