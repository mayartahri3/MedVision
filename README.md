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
| 2. Model | 4 Training · 5 Evaluation & calibration · 6 Grad-CAM | ⏳ |
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
