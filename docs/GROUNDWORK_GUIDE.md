# Groundwork Guide — what was built in T01–T05 and how to run it on Windows

> **Purpose.** This document is the user-facing companion to the Groundwork
> phase (T01–T05, ~16 hours of Day-1 + half-Day-2 work). It explains
> *what each task delivers*, *why it exists*, and *how to reproduce every
> sandbox-side check on a Windows machine*.
>
> **Read order:** Section 1 (what was built) → Section 2 (why) → Section 3
> (how to run end-to-end on your machine).

---

## 1. What was built in groundwork

The groundwork phase produced **everything below `data/raw/`** that is needed
before the modeling phase (T11+) can start. Concretely, it shipped:

### 1.1 Project scaffold (T02)
A complete directory tree (`Raman-Physics-AI-v2/`) with empty stubs in every
module, plus root files: `README.md`, `LICENSE` (MIT), `.gitignore`,
`requirements.txt`, `setup.py`, `pytest.ini`, `.python-version`.

### 1.2 Configuration files (T02 + T03 + T04)
Four YAML files in `configs/` that are the single source of truth for every
hyperparameter, path, seed, and compound name in the project:

* **`default.yaml`** — global paths, compound list (Ala/Asn/Asp/Glu/His/GlcN),
  laser wavelength (784.815734863281 nm), seed (42).
* **`data_config.yaml`** — split scheme (`"all"` runs A+A'+B side by side),
  augmentation knobs, preprocessing hyperparameters, `include_pure_in_split`
  flag.
* **`model_config.yaml`** — backbone, heads, reconstruction, MC Dropout
  (used by modeling phase, not yet read by groundwork code).
* **`train_config.yaml`** — optimizer, scheduler, loss weights, logging
  (used by modeling phase).

### 1.3 Dataloader and splits (T03)
* **`src/data/dataloader.py`** — `RawSpectraTable`, `load_raw_csv()`,
  `RamanDataset`, `build_dataloaders()`. Reads `data/raw/data.csv` (4378×1031),
  detects spectral columns automatically, converts wavelength→wavenumber
  via `(1/λ_laser − 1/λ_sample) × 10⁷`, returns PyTorch tensors of shape
  `(B, 1, 1024)` for spectra and `(B, 6)` for labels. Auto-handles the edge
  case where vial `a12` has 88 spectra instead of 90.
* **`src/data/splits.py`** — three reproducible split schemes:
  * **A — composition-level OOD** (default per project spec): 42/6/6 vials.
  * **A' — sample-level random**: 60/20/20 (less rigorous, more data).
  * **B — component-level OOD**: hold out every vial containing one
    compound. *Not useful on this dataset* — every compound appears in 49/54
    vials, so the train pool collapses to ~50 spectra. Documented as a
    stress-test only.
  * Plus `select_split()`, `build_all_splits()`, JSON save/load.
* **`scripts/prepare_data.py`** — CLI script that loads the CSV, builds the
  three splits, and writes them to `data/splits/split_{A,A_prime,B}.json`.
* **`tests/test_data.py`** — pytest suite (~25 tests) exercising helpers,
  loading invariants, dataset shapes, split correctness, JSON round-trip.

### 1.4 Classical preprocessing (T04)
* **`src/data/preprocess.py`** — four independent step functions
  (`remove_cosmic_rays`, `apply_asls_correction`, `savitzky_golay`,
  `snv_normalize`) plus the composite `preprocess_pipeline()` and a sanity
  plot helper `plot_preprocessing_steps()`.
* The pipeline is **fixed** (not learned) and **deterministic**: given the
  same input, the output is bit-identical across runs. This enforces train /
  inference parity.
* **`scripts/prepare_data.py` (extended)** — now also applies the pipeline
  to all 4378 spectra and caches the result to:
  * `data/processed/spectra_full.pt` (or `.npy` if torch absent) — `(4378, 1024) float32`
  * `data/processed/labels.pt` — `(4378, 6) float32`
  * `data/processed/wavenumbers.npy` — `(1024,) float64`
  * `data/processed/vial_ids.npy` — `(4378,) string`
  * `data/processed/preprocess_meta.json` — exact hyperparameters used.
* `tests/test_data.py` extended with `TestPreprocessing` (12 more tests).
* **`results/sanity/preprocessed_examples.png`** — visual proof: 5 real
  spectra × 5 pipeline stages.

### 1.5 Symbolic bond mapper (T05)
* **`engine/bond_mapping.json`** — 30-entry seed database mapping each
  characteristic Raman peak (cm⁻¹) to a chemical bond, vibrational mode,
  associated compound(s), and discriminative status. Includes the four
  Histidine fingerprint peaks (1003, 1180, 1495, 1575) and both Glucosamine
  pyranose peaks (1080, 1100) called out in the project instructions.
* **`engine/symbolic_mapper.py`** — `BondEntry` dataclass and `BondMapper`
  class. Methods: `from_json()`, `match_peak(wn, intensity, tolerance)`,
  `match_peaks(wns)`, `get_compound_fingerprint(compound)`, `lookup_by_id()`,
  `validate_db()`. The DB is **read-only after load**: edit the JSON and
  re-instantiate to refresh; no retraining required.
* **`tests/test_engine.py`** — pytest suite (18 tests) including
  parametrized coverage of every required project-spec peak.

### 1.6 SOTA review (T01)
* **`docs/sota_review.md`** — 213-line audit of the prior work
  (HNKHSV.pptx + the 10 `model0X.py` implementations) and articulation of
  the precise gap this MVP fills.

### 1.7 What was *not* implemented in groundwork
Reserved for later phases per the original 14-day plan:
* Architecture — `src/models/backbone.py` (ResNet-1D), `heads.py`,
  `reconstruction.py`, `uncertainty.py` → **Modeling phase (T11–T14)**
* Training loop, losses, callbacks → **Training phase (T15–T17)**
* OOD score, peak extraction, novelty locator, predict() → **Inference phase (T18–T22)**
* Augmentation body → can be combined with T11
* `engine/reference_spectra.npy` (mean pure spectra (6,1024) for the
  reconstruction module) → tiny script, will be added when T13 needs it
* Baselines (PCA+SVM, ResNet-only) → **Eval phase (T26)**

---

## 2. Why each piece exists (one-line rationale)

| Piece | Purpose | What breaks without it |
|---|---|---|
| `configs/default.yaml` | Single source of truth for compound order, seed, paths | Compound order drift between data, model, and inference |
| `dataloader.py` | Convert raw CSV into PyTorch tensors of fixed shape | Models cannot consume raw data |
| `splits.py` (Scheme A) | Vial-level holdout = OOD evaluation | Random split inflates R² by ~30 pts (proven by prior work) |
| `preprocess.py` | Deterministic pipeline = train/inference parity | Distribution shift between training and serving |
| `bond_mapping.json` + `symbolic_mapper.py` | Editable, inspectable peak→bond layer | Interpretability becomes post-hoc theatre (M07/M08 in prior work) |
| `tests/` | ~55 tests pinning current behavior | Future refactors silently break things |
| `docs/sota_review.md` | Justifies the architectural choices | Thesis defense has no benchmark to anchor against |

---

## 3. Reproducing the groundwork on your Windows machine

This section is a step-by-step recipe to take a freshly extracted T02+T03+T04+T05
repo on Windows 11 and reach the *exact* state the sandbox reached. Work in
PowerShell unless noted otherwise.

### 3.1 One-time environment setup

```powershell
cd E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2

# Use Python 3.10 (3.11 should also work; 3.12 has been observed to work too)
python --version

# Create and activate a venv
python -m venv .venv
.venv\Scripts\Activate.ps1
# If the activation script is blocked by ExecutionPolicy:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

> **CUDA users:** install PyTorch with the matching CUDA wheel BEFORE
> `pip install -r requirements.txt`. See https://pytorch.org/get-started/locally/.
> Example for CUDA 12.1:
> ```powershell
> pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
> ```

```powershell
# Install everything else
pip install --upgrade pip
pip install -r requirements.txt

# Sanity check imports
python -c "import torch, numpy, pandas, scipy, matplotlib, sklearn, pybaselines, lmfit, yaml; print('OK')"
```

### 3.2 Drop the dataset in place

```powershell
# data/raw/data.csv must exist (it's gitignored)
# Copy your file in:
copy <path-to-your-data.csv> data\raw\data.csv

# Verify
Get-Item data\raw\data.csv | Select-Object Name, Length
# Expect ~24 MB
```

### 3.3 Smoke-test the dataloader (sandbox check #1 reproduction)

```powershell
python -m src.data.dataloader
```

Expected output (last lines):
```
[INFO]   shape=(4378, 1024), wavelength range 801.62–931.28 nm, wavenumber range 267.1–2003.9 cm⁻¹
[INFO]   unique vials = 54
[INFO]   pure-vial rows detected: 60
[INFO] [train] batch shapes: spectrum=(64, 1, 1024) label=(64, 6) vial_ids=64
[INFO] [val]   batch shapes: spectrum=(64, 1, 1024) label=(64, 6) vial_ids=64
[INFO] [test]  batch shapes: spectrum=(64, 1, 1024) label=(64, 6) vial_ids=64
[INFO] ✓ Smoke test PASSED.
```

This confirms: CSV reads correctly, wavelengths convert to wavenumbers,
60 pure rows are detected, batches have the documented shapes, no NaN.

### 3.4 Run the full prepare_data pipeline (sandbox check #2 reproduction)

```powershell
python scripts\prepare_data.py --scheme all
```

Expected wall-clock: **45–75 seconds** depending on CPU. Expected log tail:

```
[INFO] Step 4: apply classical preprocessing to all 4378 spectra (cosmic → AsLS → SG → SNV)
[INFO]   pipeline kwargs: {'cosmic_threshold': 5.0, 'asls_lam': 100000.0, ...}
[INFO]   done in 43.2s  (~9.9 ms/spectrum)
[INFO] Step 5: cache preprocessed tensors to data/processed/
[INFO]   ✓ saved data/processed/spectra_full.pt (~17 MB)
[INFO]   ✓ saved data/processed/labels.pt
[INFO]   ✓ saved data/processed/wavenumbers.npy
[INFO]   ✓ saved data/processed/vial_ids.npy
[INFO]   ✓ saved data/processed/preprocess_meta.json
[INFO] Done.
```

Verify outputs:
 
```powershell
Get-ChildItem data\splits\
# Expect: split_A.json, split_A_prime.json, split_B.json (~42 KB each)

Get-ChildItem data\processed\
# Expect: spectra_full.pt, labels.pt, wavenumbers.npy, vial_ids.npy, preprocess_meta.json
```

Inspect the cache integrity:

```powershell
python -c "import torch, numpy as np; s = torch.load('data/processed/spectra_full.pt'); print(f'shape={tuple(s.shape)}, dtype={s.dtype}, mean.row.range=[{s.mean(dim=1).min():.2e}, {s.mean(dim=1).max():.2e}], std.row.range=[{s.std(dim=1).min():.4f}, {s.std(dim=1).max():.4f}], NaN={torch.isnan(s).any().item()}')"
```

Expected:
```
shape=(4378, 1024), dtype=torch.float32,
mean.row.range=[~-2.6e-7, ~2.4e-7],
std.row.range=[~1.0000, ~1.0000],
NaN=False
```

(Per-row mean ≈ 0, per-row std = 1 because of SNV; matches sandbox.)

### 3.5 Regenerate the sanity plot (sandbox check #3 reproduction)

```powershell
python -c "import yaml, numpy as np
from src.data.dataloader import load_raw_csv
from src.data.preprocess import plot_preprocessing_steps
defaults = yaml.safe_load(open('configs/default.yaml', encoding='utf-8'))
table = load_raw_csv('data/raw/data.csv',
                     defaults['compounds']['full_names'],
                     defaults['wavenumber']['laser_wavelength_nm'])
indices = [0, 100, 1000, 2000, 4000]
plot_preprocessing_steps(table.spectra[indices], table.wavenumbers,
                          'results/sanity/preprocessed_examples.png',
                          titles=[f'{table.vial_ids[i]} (#{i})' for i in indices],
                          show_intermediate=True)"
```

Open `results/sanity/preprocessed_examples.png`. You should see a 5×5 grid:
* Rows: 5 representative spectra (a01, a04, a34, a19, a38)
* Columns: Raw → Cosmic-removed → Baseline-corrected → SG-smoothed → SNV-normalized
* The Histidine fingerprint at ~1003 cm⁻¹ should be visible across all 5 rows.

### 3.6 Interactively probe the BondMapper (sandbox check #4 reproduction)

```powershell
python -c "from engine.symbolic_mapper import BondMapper
m = BondMapper.from_json('engine/bond_mapping.json')
print(m)
print('--- match 1003 ---')
for h in m.match_peak(1003): print(' ', h.id, h.bond, '->', h.compounds)
print('--- Histidine fingerprint ---')
fp = m.get_compound_fingerprint('Histidine')
print(' ', sorted(e['wavenumber_cm_inv'] for e in fp['discriminative']))
print('--- DB validation ---')
print(m.validate_db())"
```

Expected (last block):
```
BondMapper(n_entries=30, schema='1.0', compounds=['Alanine', 'Asparagine', 'Aspartic Acid', 'Glucosamine', 'Glutamic Acid', 'Histidine'])
--- match 1003 ---
  P004 Imidazole ring breathing -> ('Histidine',)
--- Histidine fingerprint ---
  [770.0, 1003.0, 1180.0, 1320.0, 1495.0, 1575.0]
--- DB validation ---
{'ok': True, 'n_entries': 30, 'n_problems': 0, 'problems': []}
```

### 3.7 Run the full test suite

```powershell
pytest tests\test_data.py tests\test_engine.py -v
```

Expected: ~55 tests pass (a few may show as `SKIPPED` if torch is unavailable
or `data/raw/data.csv` is missing). Look for the green "passed" line at the
bottom; no red "failed" anywhere.

### 3.8 Cleanup / starting fresh

If you ever want to reset and re-run from scratch:

```powershell
# Wipe processed cache and splits (raw stays)
Remove-Item -Recurse -Force data\processed\*.pt, data\processed\*.npy, data\processed\*.json -ErrorAction SilentlyContinue
Remove-Item -Force data\splits\split_*.json -ErrorAction SilentlyContinue

# Re-run from step 3.4
python scripts\prepare_data.py --scheme all
```

---

## 4. Common pitfalls and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `CSV not found: data/raw/data.csv` | Dataset not yet copied | See 3.2 |
| `ModuleNotFoundError: No module named 'torch'` | requirements not installed in active venv | `pip install -r requirements.txt` after `Activate.ps1` |
| `Windows fatal exception: code 0xc0000409` during DataLoader iter | Multi-process workers misbehaving | Set `dataloader.num_workers: 0` in `configs/data_config.yaml` |
| `Permission denied: data/raw/data.csv` | OneDrive sync lock | Pause OneDrive or move project outside synced folders |
| `RuntimeError: Found dtype Long but expected Float` | Stale processed cache from a different code version | Wipe `data/processed/` (see 3.8) and re-run prepare_data |
| `pip install` fails on `lmfit` or `pybaselines` with build error | Missing C++ build tools | Install Microsoft C++ Build Tools, then retry |

---

*Document version 1.0 — written at the end of T01 (Day-1 Groundwork).*
