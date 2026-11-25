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
| 2. Model | 4 Training · 5 Evaluation & calibration · 6 Grad-CAM, multi-model experiments | ✅ baseline trained (test AUROC 0.8075); experiments ready |
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
src/cxr/tracking.py        MLflow tracing: run per experiment, tags, params, dataset, system metrics, model
scripts/link_phase1.py     attach Phase 1 output (splits + cache) in a new Kaggle session
configs/experiments.yaml   model experiments (overrides of config.yaml)
src/cxr/evaluation.py      evaluation core: TTA, calibration, uncertainty band, report, figures
scripts/train.py           Step 4 - train one experiment (cosine/EMA/label smoothing), full MLflow trace
scripts/evaluate.py        Step 5 - test evaluation logged into the experiment's run
scripts/gradcam_eval.py    Step 6 - pointing game on NIH boxes, logged into the same run
scripts/ensemble.py        ensemble of evaluated experiments (from MLflow, no GPU)
scripts/compare_runs.py    leaderboard + charts as a "model-comparison" MLflow run
notebooks/03_experiments_kaggle.ipynb   one experiment per Kaggle commit (GPU)
notebooks/04_compare_and_ensemble.ipynb ensemble + comparison (CPU)
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

## Phase 2: model experiments (Steps 4–6)

Several models are trained and compared. Every experiment is defined in
`configs/experiments.yaml` and becomes **one MLflow run** holding its whole trace.

| Experiment | Purpose |
|---|---|
| `densenet121_224` | baseline (CheXNet-style) |
| `densenet121_320` | + higher resolution + modern recipe (cosine LR with warm-up, label smoothing, EMA) |
| `convnext_tiny_320` | modern CNN backbone |
| `efficientnet_v2_s_320` | efficient backbone |
| `ensemble-…` | mean of the members' logits (`scripts/ensemble.py`, no GPU) |

**Workflow**
1. `notebooks/03_experiments_kaggle.ipynb`: set `EXPERIMENT`, then Save & Run All (one commit per model).
   It runs `link_phase1.py` → `train.py` → `evaluate.py` → `gradcam_eval.py`, all logging into the same run.
2. `notebooks/04_compare_and_ensemble.ipynb` (CPU): `ensemble.py` + `compare_runs.py` → leaderboard run.

**What each MLflow run contains**

| Category | Logged |
|---|---|
| Reproducibility | resolved config (YAML), every hyper-parameter, git commit/branch, platform, GPU, torch version, train/val dataset digests |
| Training | per-epoch train/val loss, LR, val AUROC (mean + 15 labels), step loss, GPU/CPU/RAM usage, training curves, history.csv |
| Model | best checkpoint, PyTorch model with input/output signature (optional Model Registry) |
| Evaluation | test AUROC (mean, per label, 95% patient-bootstrap CI), AUPRC, TTA vs no-TTA, calibration error raw→calibrated, decision metrics, status distribution, uncertain labels per image, PA vs AP, band-width sweep, `calibration.json`, logits |
| Explainability | Grad-CAM pointing accuracy vs centre baseline (overall and per label), energy-in-box, gallery |

**Model selection is done on validation AUROC**, never test AUROC. Test results are reported for
every model, but choosing by test score would leak the test set into the decision.

**Decision policy** (fitted on validation, per label): `p < t_low` → negative (t_low keeps 90% of
positives), `p ≥ t_high` → positive (90% specificity), otherwise uncertain. The targets are set in
`config.yaml → eval.band`. `band_sweep_val.csv` shows how wide the uncertain zone gets for each target.

**Local use (Windows, GTX 1650 4 GB):**
```powershell
$env:CXR_WORK_DIR="D:/cxr_work"
python scripts/train.py --experiment densenet121_224 --batch-size 8 --workers 2
```
Inference with any trained experiment (download `best.pt` + `calibration.json` from its MLflow run):
```python
from cxr.inference import Predictor
p = Predictor("best.pt", "calibration.json")      # applies TTA automatically if it was used in evaluation
result = p.predict("some_xray.png")               # findings + status + heatmaps
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
