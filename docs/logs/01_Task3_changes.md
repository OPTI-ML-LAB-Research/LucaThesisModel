# T03 — Changes vs T02

This delta archive contains **only the files added or modified by T03**.
Drop them on top of your existing `Raman-Physics-AI-v2/` from T02
(overwrite when prompted). Existing T02 files not listed here are unchanged.

## File-by-file

| Path | Status | What changed |
|---|---|---|
| `configs/default.yaml` | **MODIFIED** | Added `wavenumber.laser_wavelength_nm`, `source_columns_are`, `expected_num_points`. Removed misleading `start_cm_inv`/`end_cm_inv` (those headers were nm, not cm⁻¹). |
| `configs/data_config.yaml` | **MODIFIED** | Default `split.scheme` changed to `"all"`. Added `split.include_pure_in_split` flag. Added `split.B.val_frac_within_train_pool`. |
| `src/data/dataloader.py` | **NEW** (replaces stub) | `RawSpectraTable` dataclass, `load_raw_csv()`, `is_pure_vial()`, `RamanDataset`, `raman_collate`, `build_dataloaders()`, smoke test under `python -m src.data.dataloader`. |
| `src/data/splits.py` | **NEW** (replaces stub) | Scheme A (composition OOD), A' (random), B (component OOD); JSON save/load; `select_split()` and `build_all_splits()` orchestrators. |
| `scripts/prepare_data.py` | **NEW** (replaces stub) | Loads CSV → builds split(s) per config → writes `data/splits/split_{A,A_prime,B}.json`. Step 4 (preprocess cache) deferred to after T04. |
| `tests/test_data.py` | **NEW** (replaces stub) | pytest suite: helper tests, CSV loading, dataset shapes, split correctness, JSON round-trip. Tests requiring torch are marked `pytest.importorskip("torch")` so they skip cleanly without it. |

## How to apply on Windows

From PowerShell, after extracting this ZIP next to your existing project:

```powershell
# Assumes you extracted to E:\Project\KhoaLuanCourse\T03_changes
cd E:\Project\KhoaLuanCourse
Copy-Item -Recurse -Force T03_changes\* Raman-Physics-AI-v2\
```

Then verify:

```powershell
cd Raman-Physics-AI-v2
.venv\Scripts\Activate.ps1

# Should print logs and create data/splits/split_A.json + split_A_prime.json + split_B.json
python scripts\prepare_data.py --scheme all

# Smoke test the dataloader directly (requires data/raw/data.csv to be present)
python -m src.data.dataloader

# Run the test suite (will skip torch tests if torch not installed yet)
pytest tests/test_data.py -v
```

## Verification done in sandbox

Tests run with **stubbed torch** (sandbox has no torch install):

* ✓ `load_raw_csv()` on mock 4378×1031 CSV: shape, dtypes, no NaN, label simplex sums to 1.0
* ✓ Wavelength→wavenumber conversion: self-conversion at laser λ = 0; range 267–2004 cm⁻¹
* ✓ `is_pure_vial()`: 11 parametrized cases (L-/D-/DL- prefixes, bare names, mixture vials)
* ✓ Edge case `a12` correctly carries 88 spectra (not 90)
* ✓ Pure mask detects exactly 60 pure rows (6 compounds × 10)
* ✓ Scheme A: 42/6/6 vials, zero overlap, full coverage when include_pure=True
* ✓ Scheme A': 60/20/20 row-level, fractions within 0.01%
* ✓ Scheme B: zero Histidine in train+val, > 0 in test
* ✓ `include_pure=False`: pure rows correctly excluded
* ✓ JSON save/load round-trip preserves all three index arrays
* ✓ `prepare_data.py --scheme all` end-to-end produces 3 valid JSON files

Tests requiring real torch (Dataset.__getitem__, DataLoader batching) will run
when you `pip install -r requirements.txt` on Windows.

## Notes / things to watch on real data

1. **Scheme B is sensitive to mock-data artifacts.** On the synthetic CSV, ~49/54
   vials had a non-zero Histidine fraction (Dirichlet randomness), so the train
   pool collapsed to 5 vials. On the real data, expect a much larger train pool —
   inspect log output of `prepare_data.py --scheme all` after running on real data.

2. **Wavenumber axis is descending after conversion.** Column 0 = 801.62 nm →
   ~2004 cm⁻¹; column 1023 = 931.28 nm → ~267 cm⁻¹. The dataloader keeps the
   original column order so `spectra[:, i]` aligns with `wavenumbers[i]`.
   If you want ascending wavenumbers for plotting, flip both arrays — do NOT
   flip in the dataloader (would silently break model training).

3. **`include_pure_in_split: true` is the default**, per our discussion: pure
   spectra participate in train/val/test so the model learns to recognize
   pure-compound compositions like `[0,0,0,0,1,0]`. The same pure spectra are
   *also* used to build `engine/reference_spectra.npy` — that pipeline (T03+
   in `scripts/extract_pure_references.py`) is independent.
