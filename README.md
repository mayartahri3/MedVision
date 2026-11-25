# CXR Assistant: AI decision support for chest X-ray reading

> ⚠️ **Research prototype, not a medical device.** Outputs are suggestions that a
> qualified radiologist must review. Not for clinical use.

Multi-label chest X-ray classification (DenseNet / ViT), Grad-CAM explanations, LLM-drafted
preliminary reports with uncertainty warnings, a RAG medical-knowledge assistant, and a
radiologist dashboard with human validation.

See [`docs/SCOPE.md`](docs/SCOPE.md) for scope, label decisions and success criteria.

## Roadmap
| Phase | Steps | Status |
|---|---|---|
| 1. Foundations | 1 Scope & setup · 2 Data & EDA · 3 Preprocessing | ✅ code ready |
| 2. Model | 4 Training · 5 Evaluation & calibration · 6 Grad-CAM | ✅ code ready |
| 3. Generative AI | 7 Report generation · 8 RAG assistant | ⏳ |
| 4. Application | 9 FastAPI + dashboard · 10 Docker, CI, docs | ⏳ |

## Repository layout
```
configs/config.yaml        all paths, labels, split and augmentation settings
docs/SCOPE.md              scope, disclaimer, label decisions, targets
src/cxr/config.py          config loader (+ CXR_DATA_ROOT / CXR_CACHE_ROOT env overrides)
src/cxr/data/io.py         DICOM / PNG / JPG reader -> float [0,1]
src/cxr/data/labels.py     NIH label parsing, derived Lung_Opacity, pos_weight
src/cxr/data/split.py      patient-level split + leakage assertion
src/cxr/data/transforms.py MONAI pipelines (no flips), ImageNet / torchxrayvision normalisation
src/cxr/data/dataset.py    Dataset + DataLoader factory
scripts/prepare_data.py    builds data/processed/splits.csv and pos_weight.csv
scripts/resize_cache.py    1024 px -> 320 px image cache
scripts/eda.py             figures + eda_summary.md in reports/figures/
scripts/check_pipeline.py  batch sanity checks + augmentation grid
scripts/make_synthetic.py  tiny fake dataset to test everything offline
notebooks/01_phase1_kaggle.ipynb   run Phase 1 on Kaggle
src/cxr/models/factory.py  DenseNet-121 / EfficientNet-B0 / ResNet-50, checkpoint I/O
src/cxr/metrics.py         AUROC, patient-bootstrap CIs, Platt calibration, ECE, operating points
src/cxr/explain/gradcam.py hook-free Grad-CAM + heatmap overlay
src/cxr/inference.py       Predictor: image -> calibrated findings + status + heatmaps (used by API/report)
src/cxr/tracking.py        MLflow setup (DagsHub or local SQLite)
scripts/link_phase1.py     attach Phase 1 output (splits + cache) in a new Kaggle session
scripts/train.py           Step 4 - training with early stopping, AMP, MLflow
scripts/evaluate.py        Step 5 - calibration, thresholds, uncertainty bands, test report
scripts/gradcam_eval.py    Step 6 - pointing game on NIH boxes + overlay gallery
notebooks/02_phase2_kaggle.ipynb   run Phase 2 on Kaggle (GPU)
tests/                     pytest unit tests
```

## Quick start

### A. Test locally in 2 minutes (no download)
```bash
conda create -n cxr python=3.11 -y && conda activate cxr
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GTX 1650 OK
pip install -r requirements.txt
pytest -q

python scripts/make_synthetic.py --out data/synthetic --n 400
# Windows PowerShell:  $env:CXR_DATA_ROOT="data/synthetic"; $env:CXR_CACHE_ROOT="data/synthetic_cache"
# Linux/macOS:         export CXR_DATA_ROOT=data/synthetic CXR_CACHE_ROOT=data/synthetic_cache
python scripts/prepare_data.py
python scripts/resize_cache.py
python scripts/eda.py
python scripts/check_pipeline.py --subset 64 --workers 0
```

### B. Real data on Kaggle
Open `notebooks/01_phase1_kaggle.ipynb` on Kaggle, attach the dataset `nih-chest-xrays/data`,
and run all cells. Default paths in `config.yaml` already point to Kaggle.

## Phase 2: model (Steps 4–6)

Run `notebooks/02_phase2_kaggle.ipynb` on Kaggle with a GPU, attaching the NIH dataset and the
Phase 1 notebook output. Pipeline: `link_phase1.py` → `train.py` → `evaluate.py` → `gradcam_eval.py`.

**How predictions are turned into decisions** (all fitted on validation, never on test):

| Element | What it does |
|---|---|
| Platt calibration (per label) | `p = sigmoid(a·logit + b)`. Corrects the upward bias that `pos_weight` adds to every logit, so 0.30 means about 30% |
| `t_low` | catches 95% of validation positives → below it the label is **negative** |
| `t_high` | 95% specificity on validation → at or above it the label is **positive** |
| between the two | **uncertain** → the report generator (Step 7) must add a warning |

**Local (Windows, GTX 1650 4 GB):**
```powershell
$env:CXR_WORK_DIR="D:/cxr_work"        # where checkpoints / eval / explain go
python scripts/train.py --batch-size 8 --workers 2
```
To use a model trained on Kaggle locally, download `checkpoints/best.pt` and `eval/calibration.json`:
```python
from cxr.inference import Predictor
p = Predictor("best.pt", "calibration.json")
result = p.predict("some_xray.png")        # findings + heatmaps for positive/uncertain labels
```

## Data versioning (DVC) and tracking (MLflow)
```bash
dvc init
dvc add data/processed/splits.csv data/processed/pos_weight.csv
git add data/processed/*.dvc .gitignore && git commit -m "Track patient-level splits"
# optional remote, e.g. DagsHub:  dvc remote add -d origin https://dagshub.com/<user>/cxr-assistant.dvc
```
The raw 45 GB dataset is **not** versioned. It stays on Kaggle, and only the split definition is tracked,
which is enough to reproduce every experiment.

## Key design choices
- **Patient-level splits.** NIH patients have up to hundreds of follow-up images. Splitting by image leaks patients into the test set.
- **No horizontal flips.** Flipping moves the heart to the wrong side and corrupts Cardiomegaly.
- **pos_weight per label.** Positive rates range from about 0.2% (Hernia) to about 18% (Infiltration).
- **Picklable transforms.** No lambdas, so `num_workers > 0` works on Windows.

## Datasets and licences
- NIH ChestX-ray14: NIH Clinical Center; cite Wang et al., CVPR 2017.
- CheXpert: Stanford AIMI, research-use agreement.
- MIMIC-CXR: PhysioNet credentialed access.
