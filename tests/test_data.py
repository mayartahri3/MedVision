import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr.config import all_labels, load_config  # noqa: E402
from cxr.data.labels import parse_findings  # noqa: E402
from cxr.data.split import assign_splits, check_no_leakage  # noqa: E402
from cxr.data.transforms import build_transforms  # noqa: E402

CFG = load_config()
NIH = CFG["labels"]["nih"]


def _df():
    return pd.DataFrame({
        "Image Index": [f"{i}.png" for i in range(8)],
        "Patient ID": [1, 1, 2, 2, 3, 4, 5, 6],
        "Finding Labels": ["No Finding", "Pneumonia|Effusion", "Cardiomegaly", "Infiltration",
                           "No Finding", "Edema", "Mass|Nodule", "No Finding"],
    })


def test_parse_findings_multihot_and_derived():
    out = parse_findings(_df(), NIH, CFG["labels"]["derived"])
    assert out.loc[1, ["Pneumonia", "Effusion"]].tolist() == [1, 1]
    assert out.loc[1, "Lung_Opacity"] == 1          # Pneumonia -> opacity
    assert out.loc[2, "Lung_Opacity"] == 0          # Cardiomegaly is not an opacity
    assert out.loc[0, NIH].sum() == 0 and out.loc[0, "No_Finding"] == 1


def test_unknown_label_raises():
    df = _df()
    df.loc[0, "Finding Labels"] = "Covid"
    with pytest.raises(ValueError):
        parse_findings(df, NIH)


def test_split_is_patient_disjoint():
    df = pd.DataFrame({"Image Index": [f"{i}.png" for i in range(1000)],
                       "Patient ID": np.repeat(np.arange(100), 10)})
    test = set(df[df["Patient ID"] < 20]["Image Index"])
    out = assign_splits(df, test, val_fraction=0.1, seed=0)
    check_no_leakage(out)
    assert set(out["split"]) == {"train", "val", "test"}
    assert out[out.split == "val"]["Patient ID"].nunique() == 8


def test_leakage_detected():
    df = pd.DataFrame({"Patient ID": [1, 1], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        check_no_leakage(df)


@pytest.mark.parametrize("train", [True, False])
def test_transforms_shape_and_no_flip(train):
    img = np.zeros((400, 300), np.float32)
    img[:, :150] = 1.0  # bright left half
    x = build_transforms(CFG, train=train)(img)
    s = CFG["preprocess"]["image_size"]
    assert tuple(x.shape) == (3, s, s)
    # left half must stay brighter than right half (no horizontal flip)
    assert x[:, :, : s // 2].mean() > x[:, :, s // 2:].mean()


def test_label_order():
    assert all_labels(CFG)[:14] == NIH and all_labels(CFG)[-1] == "Lung_Opacity"
