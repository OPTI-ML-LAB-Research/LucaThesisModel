# Groundwork Phase — Handover Summary (v2)

> **VERSION 2 — REVISED.** Use this document at the start of every
> subsequent chat. It incorporates the multi-dataset expansion documented
> in `docs/PROJECT_REVISION_v2.md`.
>
> **What changed from v1.** Original `data.csv` is now superseded by
> `AA_Data.csv` (with 3 extra metadata columns). We additionally have
> AAM (AA + minerals), 6 ENLIGHTEN pure-compound exports, the bacteria_ID
> dataset (Ho 2019), the API pharmaceutical dataset, and a single MoS2
> spectrum for OOD demonstration. The Zarei 2023 paper (which produced AA
> and AAM) is now the formal source citation.
>
> **Status as of end-of-Groundwork:** T01–T05 complete on the original
> single-dataset assumption. **Before Phase A starts**, a 2–3 hour
> dataloader-registry refactor must happen (Section 6 below) to absorb the
> new datasets. After that, Phase A proceeds normally.

---

## 1. Files created during Groundwork

Working directory: `E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\`.

### 1.1 Project skeleton (T02)
* `README.md`, `LICENSE` (MIT), `requirements.txt`, `setup.py`, `pytest.ini`,
  `.gitignore`, `.python-version`.
* 4 YAML configs in `configs/`.
* 90 files / 34 directories total skeleton.

### 1.2 Dataloader & splits (T03)
* `src/data/dataloader.py` — `RawSpectraTable`, `load_raw_csv`,
  `RamanDataset`, `build_dataloaders`, `is_pure_vial`.
* `src/data/splits.py` — Schemes A (composition-OOD), A' (random),
  B (component-OOD).
* `tests/test_data.py` — ~25 tests.
* **PHASE-A REWORK NEEDED** — see Section 6.

### 1.3 Classical preprocessing (T04)
* `src/data/preprocess.py` — 4 step functions + `preprocess_pipeline()` +
  `is_preprocessed` flag + `make_preprocess_fn()` + `plot_preprocessing_steps()`.
* `scripts/prepare_data.py` — CLI: load → split → preprocess → cache.
* `tests/test_data.py` extended with 12 preprocessing tests.
* `results/sanity/preprocessed_examples.png` — visual proof.

### 1.4 Symbolic bond mapper (T05)
* `engine/bond_mapping.json` — 30-entry seed DB.
* `engine/symbolic_mapper.py` — `BondMapper` class.
* `tests/test_engine.py` — 18 tests.
* **MINOR ADDITION NEEDED IN PHASE A** — quartz + calcite entries (P031, P032)
  for AAM dataset; non-breaking.

### 1.5 Documentation (T01 + handover)
* `docs/sota_review.md` — 213-line audit of HNKHSV.pptx + 10 prior baselines.
* `docs/GROUNDWORK_GUIDE.md` — Windows reproduction recipe.
* `docs/GROUNDWORK_SUMMARY.md` — this file.
* `docs/PROJECT_REVISION_v2.md` — **NEW**: detailed impact analysis of
  multi-dataset expansion. **READ THIS BEFORE PHASE A.**

---

## 2. Datasets — current inventory

| Dataset | Path | Rows × Cols | Format | Task | Phase |
|---|---|---|---|---|---|
| **AA** | `data/raw/AA_Data.csv` | 4378 × 1034 | wavelength + metadata + 6 ratios | regression | **A (primary)** |
| AAM | `data/raw/AAM_Data.csv` | 12956 × 1033 | wavelength + 8 ratios (incl. quartz, calcite) | regression | B (extension) |
| 6 pure | `data/raw/pure/*.csv` | 6 × ~10 spectra | ENLIGHTEN proprietary | reference | A (for `engine/reference_spectra.npy`) |
| quartz_calcite | `data/raw/pure/50_50_quartz_calcite.csv` | ~100 spectra | ENLIGHTEN | mineral reference | B |
| bacteria_ID | `data/raw/bacteria_ID/X_*.npy, y_*.npy` | up to 60K × 1000 | preprocessed numpy | classification (30-class) | D (stretch) |
| API | `data/raw/API/API_data.csv` + xlsx | 3511 × 3278 | wavenumber CSV | classification (33 APIs) | D (stretch) |
| MoS2 | `data/raw/ood_demo/MoS2-160o-12h-ph5.txt` | 1 spectrum, 525 rows | tab-separated | OOD demo | D (stretch) |

**Dataset facts (AA = primary):**
| Property | Value |
|---|---|
| Rows | 4378 |
| Spectral columns | 1024 (wavelength headers, 801.62 → 931.28 nm) |
| Metadata columns | `file_name`, `Repitation`, `mix_method`, `vial #` |
| Mix methods | Hand-Mixed (1440), Mechanically-Mixed (1440), Hand-Mixed_Perturbed (1438), Pure (60) |
| Wavenumber range after conversion | ~267 → ~2004 cm⁻¹ (descending, fingerprint region) |
| Laser wavelength | 784.815734863281 nm (Wasatch WP-785-R-SR-LMMF) |
| Compound order (canonical) | Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine |
| Vials | 54 total: 6 pure (`DL-alanine`, `L-asparagine`, `L-aspartic-acid`, `L-glutamic-acid`, `L-histidine`, `D-glucosamine`) + 48 mixtures (`aa01`–`aa48`) |
| Spectra per vial | 10 (pure) or 90 (mixture); except `aa12` which has 88 |
| Pure rows | 60 (= 6 × 10) |
| Label simplex | Sums to 1.0 ± 1e-6 |
| Histidine fingerprint peaks | 1003, 1180, 1495, 1575 cm⁻¹ (per project spec) |
| Glucosamine fingerprint peaks | 1080, 1100 cm⁻¹ |
| Quartz peak (AAM only) | 464 cm⁻¹ |
| Calcite peak (AAM only) | 1086 cm⁻¹ |

**Source paper:** Zarei et al., *Anal. Chem.* 95(43) 15908-15916 (2023).

---

## 3. Open issues / known constraints

### Issue #1 (NOT a bug — design constraint)
**Scheme B (component-OOD) is fundamentally limited on AA.** Every compound
appears with non-zero ratio in 49/54 vials → train pool collapses to ~50 rows.
**Action:** Use Scheme A as primary OOD eval; Scheme B is documented as
stress-test only.

### Issue #2 (NEW — addressed in PROJECT_REVISION_v2)
**Original `data.csv` is superseded by `AA_Data.csv`.** They share the 1024
spectral cols and 6 ratio cols, but `AA_Data.csv` has 3 extra metadata cols
(`file_name`, `Repitation`, `mix_method`). Phase A should:
1. Replace `data/raw/data.csv` with `data/raw/AA_Data.csv`
2. Update `configs/default.yaml` registry path
3. Verify by re-running `pytest tests/test_data.py`

### Issue #3 (NEW — must be done before Phase A modeling)
**`src/data/dataloader.py` is hard-coded for one CSV format.** Section 6
below lists the refactor. ~2 hours.

### Issue #4 (NEW — Phase A unblocks this)
**`engine/reference_spectra.npy` not yet built.** Now buildable from the
6 ENLIGHTEN pure-compound CSVs via a script
(`scripts/extract_pure_references.py`). Required by reconstruction
module (T13).

---

## 4. Decisions changed vs. original plan

(Includes both T01–T05 decisions AND new revisions.)

| # | Original plan | What was actually done | Reason |
|---|---|---|---|
| 1 | Default split scheme = A only | `split.scheme: "all"` runs A + A' + B side-by-side | User requested 3-way comparison |
| 2 | `include_pure_in_split` not in plan | Added as config flag | Pure samples need to be in train AND in `engine/reference_spectra.npy` |
| 3 | Scheme B holdout = "Histidine" | Default kept, documented as not useful | Discovered 49/54 vials contain every compound |
| 4 | T01 SOTA review = first task | Deferred to last | T01 is doc-only |
| 5 | Bond DB seed file user said was uploaded | Built from scratch using project spec | File not found in uploads |
| 6 | YAML preprocessing keys had selectors (`cosmic_ray_method`) | Replaced by direct hyperparameters | Pipeline is fixed; no selector branching |
| 7 | Preprocessing on-the-fly default | `apply_on_the_fly: false` | Faster training via cache |
| 8 | Pure Alanine = `L-alanine` | Real data uses `DL-alanine` | `is_pure_vial()` regex handles all prefixes |
| 9 | `splits.py` was scheme-A-only | Implements all 3 schemes + JSON persistence | User's "run 3 splits side-by-side" request |
| 10 | `pytest.ini` not in original spec | Added | Required for pytest markers |
| 11 | Single dataset assumption | **MUST become multi-dataset registry** | 5 new datasets received after T05; see PROJECT_REVISION_v2.md |
| 12 | `data.csv` as primary input | **Switch to `AA_Data.csv`** | Superset with `mix_method` metadata |
| 13 | `engine/reference_spectra.npy` deferred to "when T13 needs it" | **Build immediately at start of Phase A** | Now we have real ENLIGHTEN refs (better than averaging training pure samples) |
| 14 | OOD evaluation only via Scheme B | **Scheme C added (mix_method holdout) + 3 free OOD test sets** | bacteria_ID, MoS2, mineral peaks in AAM all serve as real OOD samples |
| 15 | No external benchmarks | **bacteria_ID + API + MoS2 added as Day-13 stretch** | Datasets now available |

---

## 5. Cached artefacts available to Phase A

After running `python scripts/prepare_data.py --scheme all` (current behavior):

| File | Shape | Purpose |
|---|---|---|
| `data/processed/spectra_full.pt` | (4378, 1024) float32 | Preprocessed AA spectra |
| `data/processed/labels.pt` | (4378, 6) float32 | AA composition targets |
| `data/processed/wavenumbers.npy` | (1024,) float64 | Raman shift axis |
| `data/processed/vial_ids.npy` | (4378,) string | Per-row sample IDs |
| `data/processed/preprocess_meta.json` | dict | Pipeline hyperparameters used |
| `data/splits/split_A.json` | indices | Composition-OOD: 3538 / 460 / 380 |
| `data/splits/split_A_prime.json` | indices | Random: 2627 / 875 / 876 |
| `data/splits/split_B.json` | indices | Histidine holdout: 40 / 10 / 4328 (stress test) |

**WARNING:** these are built from the OLD `data.csv`. After Section 6 refactor,
re-run `prepare_data.py --dataset AA` to regenerate from `AA_Data.csv`. Files
should be similar but `vial_ids.npy` will differ if AA uses different vial
naming (verified: `AA_Data.csv` uses `aa01`-`aa48`, possibly different from
old `a01`-`a48`).

---

## 6. **MANDATORY pre-Phase-A refactor (~2-3 hours)**

Before starting any modeling task in Phase A, execute these in order:

### 6.1 File system reorganization
```powershell
cd E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\data\raw

# Backup old data.csv
Move-Item data.csv data.csv.legacy -Force

# Drop new files
mkdir pure, bacteria_ID, API, ood_demo
# Move uploaded files into the right subfolders (see PROJECT_REVISION_v2.md Section 2.9)
```

### 6.2 Code refactor checklist
1. ☐ Update `configs/default.yaml` with `datasets:` registry (PROJECT_REVISION_v2.md §2.1)
2. ☐ Refactor `src/data/dataloader.py` — split `load_raw_csv` into format-specific loaders dispatched by registry (§2.3)
3. ☐ Add `src/data/enlighten_parser.py` — parse ENLIGHTEN proprietary CSV (§2.4)
4. ☐ Add `src/data/bacteria_id_loader.py` — load Ho 2019 numpy paradigm (§2.5)
5. ☐ Implement `scripts/extract_pure_references.py` body — build `engine/reference_spectra.npy` (§2.7)
6. ☐ Add P031 (quartz 464 cm⁻¹) and P032 (calcite 1086 cm⁻¹) to `engine/bond_mapping.json` (§2.8)
7. ☐ Add `split_C_mix_method_ood()` to `src/data/splits.py` (§3.2)
8. ☐ Update `tests/test_data.py` — add tests for new loaders, keep old tests passing
9. ☐ Re-run `pytest tests/` end-to-end — must be green
10. ☐ Re-run `python scripts/prepare_data.py --dataset AA --scheme all` to rebuild cache

### 6.3 Acceptance test
After 6.2 is done, this snippet should work:
```python
from src.data.dataloader import load_dataset
import yaml
cfg = yaml.safe_load(open("configs/default.yaml"))

# Primary AA dataset
table_aa = load_dataset("AA", cfg)
assert table_aa.spectra.shape == (4378, 1024)

# AAM with 8-component labels
table_aam = load_dataset("AAM", cfg)
assert table_aam.labels.shape == (12956, 8)

# bacteria_ID (returns dict of tensors, not RawSpectraTable)
bact = load_dataset("bacteria_ID", cfg)
assert bact["X_test"].shape == (3000, 1000)
assert bact["y_test"].max() == 29

# MoS2 (returns single (P,) array)
mos2 = load_dataset("MoS2", cfg)
assert mos2.spectra.shape == (1, 525)
```

---

## 7. Three-tier training plan (full project)

| Tier | When | Datasets | Model | Goal |
|---|---|---|---|---|
| 1 | Phase A (Day 3-7) | AA only | 6-output regression | Hit MAE ≤ 0.020 on Scheme A test set |
| 2 | Phase B (Day 8-10) if T1 ahead | AA + AAM | 6 or 8-output regression | Mineral robustness; 30% extra data |
| 3 | Phase D (Day 13) stretch | + bacteria_ID + API + MoS2 (frozen Tier-2 model) | OOD detector evaluation | Show MVP correctly flags cross-domain OOD |

---

## 8. Quick-start for Phase A (after Section 6 refactor)

```python
# Loading the primary AA dataset:
from src.data.dataloader import load_dataset
from src.data.splits import load_split_from_json
import torch, yaml, numpy as np

cfg = yaml.safe_load(open("configs/default.yaml"))
table = load_dataset("AA", cfg)                # RawSpectraTable
split = load_split_from_json("data/splits/split_A.json")

# Or, faster: load from preprocessed cache
spectra = torch.load("data/processed/spectra_full.pt")     # (4378, 1024)
labels  = torch.load("data/processed/labels.pt")           # (4378, 6)

# Pure references for reconstruction module (T13):
import numpy as np
refs = np.load("engine/reference_spectra.npy")             # (6, 1024)
# (Built by scripts/extract_pure_references.py from ENLIGHTEN exports)

# Bond lookup (works unchanged):
from engine.symbolic_mapper import BondMapper
mapper = BondMapper.from_json("engine/bond_mapping.json")
hits = mapper.match_peak(1003)                              # → P004 Histidine
```

---

## 9. Read order for next chat

1. **This file** (`GROUNDWORK_SUMMARY.md`) — overview, what's done, what's open.
2. **`PROJECT_REVISION_v2.md`** — detailed multi-dataset impact analysis,
   read in full before Phase A coding.
3. **`sota_review.md`** — for thesis-defense context (not Phase-A-specific).
4. **`GROUNDWORK_GUIDE.md`** — only if running the build from scratch.

---

*Generated at the end of T01 (last task of Groundwork phase), then updated
with multi-dataset revision before Phase A. Document version 2.0.*
