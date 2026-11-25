# CXR Assistant: Project Scope (Step 1)

## Purpose
This is a **research and clinical decision-support prototype**. It helps radiologists by
flagging suspicious findings on chest X-rays, showing where the model looked (Grad-CAM),
drafting a preliminary report and answering questions from trusted references.

> **It is not a diagnostic device.** Every output is a suggestion that a qualified
> professional must review. The model was trained on public research data with
> noisy labels and has not been clinically validated.
> This disclaimer appears in the UI, in every generated report, in the API responses and in the README.

## In scope
| Area | Decision |
|---|---|
| Modality | Chest X-ray, frontal views only (PA and AP) |
| Input formats | DICOM, PNG, JPG |
| Primary dataset | NIH ChestX-ray14 (112,120 images, 30,805 patients, 14 labels) |
| Later datasets | CheXpert (native *Lung Opacity* label; needs registration), MIMIC-CXR (images + reports; needs PhysioNet credentialing) |
| Task | Multi-label classification (an image can have several findings) |
| Trained labels | All 14 NIH findings + derived `Lung_Opacity` |
| Focus labels (UI / report) | Pneumonia, Cardiomegaly, Lung Opacity, Effusion |
| Explainability | Grad-CAM per predicted class |
| Human-in-the-loop | Radiologist approves, edits or rejects every report; feedback is logged |

## Out of scope
Lateral views, CT/MRI, pediatric-specific models, segmentation, real patient data, and
any use for actual clinical decisions.

## Label decisions
- **Lung_Opacity is derived.** NIH has no such label. We set it positive when any of
  Infiltration, Consolidation, Pneumonia, Edema or Atelectasis is present. This only
  approximates CheXpert's definition and should be replaced by the native label once
  CheXpert is added.
- **"No Finding" is not a training target.** It is the all-zero vector and is kept as a column for analysis.
- **Known label noise.** NIH labels were text-mined from reports. Expect errors,
  especially for Pneumonia, Infiltration and Consolidation, which overlap a lot.

## Data-split rules
1. Test set = official NIH `test_list.txt` (patient-disjoint).
2. Validation = 10% of the remaining **patients** (never split by image).
3. `check_no_leakage()` is asserted every time splits are built.

## Success criteria (targets for Steps 4–5)
| Metric | Target |
|---|---|
| Mean AUROC (14 labels, official test set) | ≥ 0.80 (published DenseNet-121 baselines report about 0.80–0.84) |
| Cardiomegaly AUROC | ≥ 0.85 |
| Calibration (ECE after Platt scaling on val, measured on test) | ≤ 0.05 |
| Grad-CAM pointing game on NIH bounding boxes | reported (no fixed target) |
| Every report contains only findings present in the model output | 100% (automated check) |

## Compute plan
- Training: Kaggle GPU (NIH is available as the Kaggle dataset `nih-chest-xrays/data`).
- Local (GTX 1650, 4 GB): development, debugging, inference, demo app.
- Images are cached at 320 px once (`resize_cache.py`); training uses 224 px.

## Risks
| Risk | Mitigation |
|---|---|
| Patient leakage inflates metrics | Patient-level split + assertion |
| Class imbalance (Hernia ≈ 0.2%) | `pos_weight` in BCE, per-class thresholds, AUROC instead of accuracy |
| AP/PA confounding | Report metrics per view (Step 5) |
| LLM hallucinated findings | Structured JSON input + post-generation check (Step 7) |
| Over-trust by users | Calibrated scores, uncertainty flags, mandatory human sign-off |
