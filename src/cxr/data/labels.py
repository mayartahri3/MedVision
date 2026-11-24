"""Parse NIH 'Finding Labels' strings into a multi-hot table."""
from __future__ import annotations

import pandas as pd


def parse_findings(df: pd.DataFrame, nih_labels: list[str],
                   derived: dict[str, list[str]] | None = None,
                   col: str = "Finding Labels") -> pd.DataFrame:
    """Add one 0/1 column per label.

    NIH encodes labels as 'Effusion|Infiltration' or 'No Finding'.
    'No Finding' rows simply get all zeros.
    """
    out = df.copy()
    findings = out[col].fillna("").str.split("|")
    for lab in nih_labels:
        out[lab] = findings.apply(lambda fs, l=lab: int(l in fs)).astype("int8")

    unknown = set(f for fs in findings for f in fs) - set(nih_labels) - {"No Finding", ""}
    if unknown:
        raise ValueError(f"Unexpected labels in metadata: {sorted(unknown)}")

    for name, members in (derived or {}).items():
        out[name] = out[members].max(axis=1).astype("int8")

    out["No_Finding"] = (out[nih_labels].sum(axis=1) == 0).astype("int8")
    return out


def positive_weights(df: pd.DataFrame, labels: list[str]) -> pd.Series:
    """neg/pos ratio per label, for BCEWithLogitsLoss(pos_weight=...) in Step 4."""
    pos = df[labels].sum()
    neg = len(df) - pos
    return (neg / pos.clip(lower=1)).rename("pos_weight")
