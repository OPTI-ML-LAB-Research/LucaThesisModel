# Project Directory Structure — Raman Physics-Informed AI MVP

```
Raman-Physics-AI/
│
├── README.md                          # Project overview, quickstart, citation
├── LICENSE                            # MIT or Apache-2.0 recommended
├── requirements.txt                   # pinned dependencies
├── setup.py                           # optional: makes project pip-installable
├── .gitignore                         # ignore: __pycache__, .venv, checkpoints/, results/large_files
├── .python-version                    # 3.10 recommended
│
├── configs/                           # YAML/JSON configuration files
│   ├── default.yaml                   # default hyperparameters
│   ├── train_config.yaml              # training-specific (epochs, lr, etc.)
│   ├── model_config.yaml              # architecture choices
│   └── data_config.yaml               # paths, split strategies
│
├── data/                              # raw and processed data (gitignored if large)
│   ├── raw/
│   │   └── data.csv                   # original 4378 × 1031 CSV from old project
│   ├── processed/                     # cached preprocessed tensors
│   │   ├── train.pt
│   │   ├── val.pt
│   │   └── test.pt
│   ├── splits/                        # JSON files defining splits
│   │   ├── split_A_composition_ood.json
│   │   └── split_B_component_ood.json
│   └── reference/                     # pure reference spectra
│       ├── alanine_mean.npy
│       ├── glycine_mean.npy
│       └── ...
│
├── docs/                              # documentation, theory, reports
│   ├── REPORT.md                      # main methodology report (English)
│   ├── sota_review.md                 # Day-1 gap analysis
│   ├── theory/
│   │   ├── physics_loss_derivation.md # math behind reconstruction loss
│   │   ├── ood_score_design.md        # rationale for OOD scoring
│   │   └── bond_mapping_methodology.md # how DB was constructed
│   ├── thesis_chapter.md              # for graduation defense (Vietnamese OK)
│   └── figures/                       # publication-quality figures
│       ├── architecture_diagram.png
│       ├── workflow.png
│       └── benchmark_chart.png
│
├── engine/                            # domain knowledge & symbolic modules (NOT learned)
│   ├── __init__.py
│   ├── bond_mapping.json              # the seed DB (provided)
│   ├── reference_spectra.npy          # (6, 1024) tensor of mean pure spectra
│   ├── peak_extractor.py              # find_peaks + Voigt fitting
│   ├── symbolic_mapper.py             # peak → bond mapping lookup
│   ├── novelty_locator.py             # detect peaks not in DB
│   └── physics_constraints.py         # Beer-Lambert linearity check, etc.
│
├── src/                               # all learnable code
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py              # PyTorch Dataset & DataLoader (port from old)
│   │   ├── preprocess.py              # classical pipeline: AsymLS + cosmic + SG + SNV
│   │   ├── augmentation.py            # shift/intensity/noise during training
│   │   └── splits.py                  # split_A() and split_B() functions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py                # 1D-ResNet feature extractor
│   │   ├── heads.py                   # quantification head (softmax simplex)
│   │   ├── reconstruction.py          # s_recon = Σ(α_i · scale_i · pure_i)
│   │   ├── uncertainty.py             # MC Dropout wrapper
│   │   ├── full_model.py              # ties backbone + heads together
│   │   └── baselines/
│   │       ├── __init__.py
│   │       ├── pca_svm.py             # baseline 1
│   │       ├── resnet_only.py         # baseline 2 (no physics)
│   │       └── nmf_plsr.py            # optional: from old thesis
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py                  # MAE, physics_loss, combined_loss
│   │   ├── train.py                   # main training loop
│   │   ├── trainer_class.py           # OO-style alternative
│   │   └── callbacks.py               # checkpointing, early stopping, logging
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py                 # main predict(spectrum) function
│   │   ├── ood.py                     # OOD scoring
│   │   ├── report.py                  # JSON + Markdown report generation
│   │   └── visualize.py               # plotting utilities
│   │
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py                 # ID Acc, MAE, AUROC, CVR, etc.
│       ├── benchmark.py               # runs all baselines + our model
│       └── compare.py                 # generates comparison table
│
├── tests/                             # pytest test suite
│   ├── test_data.py                   # dataloader, preprocess
│   ├── test_models.py                 # forward pass shapes, no NaN
│   ├── test_engine.py                 # peak extraction, symbolic mapping
│   ├── test_inference.py              # predict() returns valid dict
│   └── test_metrics.py                # metric correctness on known cases
│
├── notebooks/                         # exploratory analysis (not committed for production)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_sanity.ipynb
│   ├── 03_model_debugging.ipynb
│   ├── 04_results_analysis.ipynb
│   └── 05_demo_walkthrough.ipynb      # for thesis defense demonstration
│
├── checkpoints/                       # saved model weights (gitignored)
│   ├── best.pt
│   ├── last.pt
│   └── baselines/
│       ├── pca_svm.pkl
│       └── resnet_only_best.pt
│
├── results/                           # all experimental outputs
│   ├── training_log.csv               # per-epoch metrics
│   ├── benchmark_table.md             # comparison table
│   ├── benchmark_table.csv
│   ├── reports/                       # per-sample analysis reports
│   │   ├── demo_alanine_glycine_50_50.md
│   │   ├── demo_histidine_pure.md
│   │   └── demo_ood_unknown.md
│   ├── figures/                       # all plots, organized by experiment
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── ood_roc.png
│   │   └── reconstruction_examples/
│   ├── sanity/                        # Day-2/3 sanity check outputs
│   │   ├── pure_spectra.png
│   │   ├── preprocessed_examples.png
│   │   └── peak_demo_*.png
│   └── midcheckpoint_report.md        # Day-7 GO/NO-GO decision
│
├── dashboard/                         # Streamlit UI (optional, Day 13)
│   ├── app.py                         # main Streamlit entry
│   ├── components/
│   │   ├── upload.py
│   │   ├── analysis_view.py
│   │   └── report_view.py
│   └── assets/                        # logos, sample spectra for demo
│
├── scripts/                           # one-off utility scripts
│   ├── prepare_data.py                # one-time: builds processed/ from raw/
│   ├── extract_pure_references.py     # one-time: builds reference/ from raw/
│   ├── train_all_baselines.sh         # batch train all baselines
│   ├── reproduce_results.sh           # full reproduction pipeline
│   └── export_for_paper.py            # generate figures + tables for paper
│
└── benchmark/                         # external test sets (Day 13 stretch)
    ├── bacteria_id/                   # Bacteria-ID dataset (if downloaded)
    ├── rruff/                         # RRUFF subset (if downloaded)
    └── benchmark_runner.py            # runs our model on external data
```

---

## Folder-by-folder rationale

### `configs/` — Centralize hyperparameters
Don't bury hyperparameters in code. YAML files mean you can swap configs without code changes:
```bash
python src/training/train.py --config configs/train_config.yaml
```

### `data/raw/` vs `data/processed/`
Raw is sacred (never modified). Processed is regenerable from `scripts/prepare_data.py`. This separation lets you experiment with different preprocessing without losing the original.

### `engine/` — The "non-learned" knowledge
This is **the differentiator** of your project vs. pure-DL baselines. Bond mapping, peak extraction, physics constraints — all **deterministic, inspectable, modifiable**. When defending the thesis, point at this folder: "This is what makes my model interpretable, not black-box."

### `src/` — All trainable code
Separated into 5 sub-modules with clear responsibilities. Resists "god class" anti-pattern.

### `tests/` — Don't skip this
Even minimal tests (5-10) catch 80% of regressions. When you change loss function on Day 9 and accidentally break Day 6 functionality, tests scream immediately.

### `notebooks/` — For exploration, not production
Notebooks are for thinking. When something works, **port to `src/`**. Don't ship notebooks as the model.

### `results/` — Append-only, structured
Don't overwrite. Each experiment gets a timestamped subfolder if needed. By Day 14, this folder is your thesis evidence.

---

## Quickstart commands (after setup)

```bash
# One-time setup
git clone <your-repo-url>
cd Raman-Physics-AI
python -m venv .venv
source .venv/bin/activate          # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python scripts/prepare_data.py     # builds data/processed/

# Training
python src/training/train.py --config configs/train_config.yaml

# Single-spectrum inference
python -c "
from src.inference.predict import predict
import numpy as np
spectrum = np.load('data/raw/example.npy')
report = predict(spectrum)
print(report['composition'])
"

# Reproduce all benchmarks
bash scripts/reproduce_results.sh

# (Optional) Launch dashboard
streamlit run dashboard/app.py
```

---

## What goes in `requirements.txt`

```
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Spectroscopy
lmfit>=1.2.0                          # Voigt fitting
pybaselines>=1.0.0                    # asymmetric LS baseline (alternative: airPLS)

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0                        # for interactive plots

# Experiment tracking
wandb>=0.15.0                         # or tensorboard
pyyaml>=6.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Optional
streamlit>=1.28.0                     # only if doing T28
ipykernel>=6.25.0                     # for notebooks
tqdm>=4.65.0
```

---

## Initial `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.python-version

# Data (large files)
data/raw/*.csv
data/processed/*.pt
checkpoints/*.pt
checkpoints/*.pkl

# Results (selective: commit small files, gitignore large ones)
results/figures/*.png        # comment out if you want figures in repo
results/sanity/*.png

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb_meta

# Logs
*.log
wandb/
runs/
```

---

## Critical rule: One source of truth per concept

| Concept | Lives in | Do NOT also live in |
|---|---|---|
| Reference pure spectra | `engine/reference_spectra.npy` | scattered across notebooks |
| Bond mapping | `engine/bond_mapping.json` | hardcoded in Python |
| Hyperparameters | `configs/*.yaml` | scattered as default args |
| Train/val/test indices | `data/splits/*.json` | recomputed each run |
| Model architecture choice | `configs/model_config.yaml` | command-line flags |

When a value matters (changes results), it goes in a config file or a versioned data file. Never in print statements, comments, or "I'll remember it."
