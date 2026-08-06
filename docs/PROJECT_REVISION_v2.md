# Project Revision v2 — Impact of New Datasets on T01–T05 + Phase-A Plan

> **Purpose.** Document everything that needs to change in the existing project
> (instructions, folder layout, configs, code, training strategy) now that we
> have access to **5 additional datasets** beyond the original `data.csv`.
>
> **Status.** Written between Groundwork (T01–T05) and Phase A (modeling).
> No code has been changed yet; this document specifies what *should* change
> and why. Phase A will implement the changes.

---

## 1. New datasets — what we now have

| Dataset | Source | Size | Format | Task | Role in MVP |
|---|---|---|---|---|---|
| **AA** (`AA_Data.csv`) | Zarei 2023 paper, primary | 24 MB, 4378×1034 | wavelength columns + `mix_method`, `vial #`, 6 ratios | regression (composition) | **PRIMARY** — replaces old `data.csv` |
| **AAM** (`AAM_Data.csv`) | Zarei 2023 paper, AA+minerals | 91 MB, 12956×1033 | wavelength + `names` + 8 ratios (6 AA + quartz + calcite) | regression (composition + minerals) | **EXTENSION** — Phase B+ stretch |
| **6 pure compounds** (`DL-alanine.csv` etc.) | ENLIGHTEN spectrometer raw exports | 6 × ~185 KB | proprietary header + 1024 pixels × ~10 spectra/file | reference spectra | **REFERENCE** — for `engine/reference_spectra.npy` |
| **`50_50_quartz_calcite.csv`** | ENLIGHTEN raw, mineral background | 1.7 MB, ~100 spectra | same as above | mineral-only reference | **REFERENCE** — for AAM reconstruction |
| **bacteria_ID** (`X_*.npy`, `y_*.npy`, `wavenumbers.npy`) | Ho et al. 2019 (Nature Comms) | 60K + 3K + 3K + 10K + 2.5K spectra × 1000 channels | preprocessed numpy, classes 0–29 | **classification (30 species)** | **EXTERNAL BENCHMARK** — Day 13 stretch |
| **API** (`API_data.csv` + `Product_information.xlsx`) | Pharmaceutical APIs, 33 chemicals | 185 MB, 3511×3278 | wavenumber columns | classification or fingerprint matching | **EXTERNAL BENCHMARK** — Day 13 stretch |
| **MoS2** (`MoS2-160o-12h-ph5.txt`) | external Raman, single spectrum | 6.8 KB, 525 rows | tab-separated `wavenumber↹intensity` | OOD demo | **OOD TEST CASE** — show novelty detection |

### 1.1 Key insight: the original `data.csv` is a *subset* of `AA_Data.csv`
Verified by column comparison: `AA_Data.csv` has 3 extra metadata columns
(`file_name`, `Repitation`, `mix_method`) that `data.csv` lacks. The 1024
spectral columns and 6 ratio columns are identical in both. **All work done
in T03–T05 ports forward unchanged**, but we should switch to `AA_Data.csv`
to gain the `mix_method` metadata (4 categories: Hand-Mixed, Mechanically-Mixed,
Hand-Mixed_Perturbed, Pure).

### 1.2 Source paper now identified
Zarei et al., *Anal. Chem.* 95(43) 15908-15916 (2023) —
*"Machine Learning Analysis of Raman Spectra to Quantify the Organic
Constituents in Complex Organic-Mineral Mixtures"*. This is the **same paper**
the prior HNKHSV thesis cited but didn't fully use. Confirms:

* Laser **784.82 nm** (matches our `laser_wavelength_nm = 784.815734863281`)
* Spectral range 270–2000 cm⁻¹ (matches detected range)
* Three datasets in the paper: AA (4320), AAF (2880, with fluorescence),
  AAM (12960, with minerals). **We have AA + AAM, missing AAF.**
* PLSR/CNN reach R² = 0.92–0.98 on these datasets in the paper, but uses
  **leave-one-out cross-validation** (not group-OOD split). Our group-OOD
  approach is more conservative.

---

## 2. Required changes — files / configs / code

### 2.1 `configs/default.yaml` — ADD multi-dataset registry

Current state lists exactly one dataset path:
```yaml
paths:
  data_raw_csv: "data/raw/data.csv"
```

**Replace with a registry** that lets training/inference pick a dataset by name:

```yaml
datasets:
  # PRIMARY dataset — the one Phase A trains on
  primary: "AA"

  # Registry of all known datasets and their format adapters
  registry:
    AA:
      path: "data/raw/AA_Data.csv"
      format: "wide_csv_wavelength"      # spectral cols are wavelengths in nm
      vial_col: "vial #"
      label_cols: [Alanine, Asparagine, "Aspartic Acid", "Glutamic Acid", Histidine, Glucosamine]
      meta_cols: [mix_method, file_name, Repitation]   # optional, kept for analysis
      task: regression
      laser_wavelength_nm: 784.815734863281

    AAM:
      path: "data/raw/AAM_Data.csv"
      format: "wide_csv_wavelength"
      vial_col: "names"                  # different from AA
      label_cols: [alanine, asparagine, aspartic_acid, glutamic_acid, histidine, glucosamine, quartz, calcite]
      task: regression                    # 8-component, includes minerals
      laser_wavelength_nm: 784.815734863281

    bacteria_ID:
      path: "data/raw/bacteria_ID/"      # directory with X_*.npy, y_*.npy
      format: "ho2019_npy"
      task: classification
      num_classes: 30
      preprocessed: true                  # already normalized to [0, 1]
      wavenumbers_file: "data/raw/bacteria_ID/wavenumbers.npy"

    API:
      path: "data/raw/API_data.csv"
      format: "wide_csv_wavenumber"      # spectral cols are ALREADY in cm⁻¹
      info_xlsx: "data/raw/Product_information.xlsx"
      task: classification

    MoS2:
      path: "data/raw/MoS2-160o-12h-ph5.txt"
      format: "two_col_txt_wavenumber"
      task: ood_demo                      # single spectrum, no labels
```

### 2.2 `configs/default.yaml` — UPDATE compound list (no change needed)

Compound names in `AA_Data.csv` use the same English names as `data.csv`
(`Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine`).
For AAM, however, the column names are **lowercase with underscores**
(`alanine, asparagine, aspartic_acid, ...`). The dataset registry above
handles this via the `label_cols` field; the dataloader reads from registry
instead of from a single global compound list.

### 2.3 `src/data/dataloader.py` — REFACTOR to handle multiple formats

**Current state (T03):** One function `load_raw_csv()` hard-coded for
`data.csv` schema (1024 spectral cols + `vial #` + 6 ratio cols).

**Required change:** Split into format-specific loaders dispatched by
`format` field:

```python
def load_dataset(name: str, registry: dict) -> RawSpectraTable:
    """Load any registered dataset by name."""
    cfg = registry["registry"][name]
    fmt = cfg["format"]
    loader = {
        "wide_csv_wavelength":   _load_wide_csv,         # AA, AAM
        "wide_csv_wavenumber":   _load_wide_csv,         # API
        "ho2019_npy":            _load_bacteria_id,      # bacteria_ID
        "two_col_txt_wavenumber": _load_two_col_txt,     # MoS2
        "enlighten_export":      _load_enlighten,        # 6 pure CSVs
    }[fmt]
    return loader(cfg)
```

The existing T03 `load_raw_csv()` body becomes the implementation of
`_load_wide_csv()` with one extra branch: if `format == 'wide_csv_wavenumber'`,
skip the wavelength→wavenumber conversion.

### 2.4 NEW module: `src/data/enlighten_parser.py`

The 6 pure-compound CSVs (`DL-alanine.csv`, `L-asparagine.csv`, etc.) are
**ENLIGHTEN spectrometer exports** — a proprietary text format with ~30
header rows of metadata and spectra arranged as **adjacent column pairs**
(spacer + intensity). Need a dedicated parser. Used by the new
`scripts/extract_pure_references.py` to build `engine/reference_spectra.npy`.

Structure (verified on `DL-alanine.csv`):
```
ENLIGHTEN Version,2.2.7
Measurement ID,,,<id1>,,<id2>,,...        (10 spectra per file)
Serial Number,,,WP-00643,,WP-00643,,...
... ~30 rows of metadata ...
Note,,,DL-alanine,,DL-alanine,,...        (compound name, repeated)
Laser Wavelength,,,784.816,,...
Pixel,Wavelength,Wavenumber,Processed,,Processed,,...    (column header)
0,801.62,267.1,1234.5,,1235.7,,...        (data starts here, ~1024 rows)
```

The parser must skip metadata, find the `Pixel,Wavelength,Wavenumber,...`
header row, and extract only the `Processed` columns. Return shape (N, 1024)
for each file.

### 2.5 NEW module: `src/data/bacteria_id_loader.py`

The bacteria_ID dataset uses Ho et al.'s **finetune paradigm**: large
`X_reference` (60K spectra) for pretraining, small `X_finetune` (3K) for
adaptation, separate `X_test` for held-out evaluation, plus two clinical
distribution-shift sets (`X_2018clinical`, `X_2019clinical`). Wavenumbers
are pre-computed in `wavenumbers.npy` (1000 channels, 381–1792 cm⁻¹).
Spectra are **already preprocessed** (range [0, 1]).

Need a 4-way loader returning (reference, finetune, test, clinical) splits
that bypasses our T04 preprocessing pipeline (already done upstream).

### 2.6 `src/data/preprocess.py` — ADD bypass when `preprocessed: true`

Trivial change: respect `cfg["preprocessed"]` flag from registry; if true,
return the input unchanged. Already partially implemented via the
`is_preprocessed` flag — just wire it to the registry.

### 2.7 `engine/reference_spectra.npy` — NOW BUILDABLE FROM REAL DATA

We previously deferred this. With the 6 pure-compound CSVs we can build it
properly:

```python
# scripts/extract_pure_references.py — implement the body
from src.data.enlighten_parser import load_enlighten_csv
import numpy as np

PURE_FILES = [
    "data/raw/pure/DL-alanine.csv",
    "data/raw/pure/L-asparagine.csv",
    "data/raw/pure/L-aspartic-acid.csv",
    "data/raw/pure/L-glutamic-acid.csv",
    "data/raw/pure/L-histidine.csv",
    "data/raw/pure/D-glucosamine-HCl.csv",
]
refs = []
for f in PURE_FILES:
    spectra = load_enlighten_csv(f)            # (N, 1024)
    refs.append(spectra.mean(axis=0))           # (1024,) mean spectrum
ref_tensor = np.stack(refs)                     # (6, 1024)
# Apply the same preprocessing pipeline used in training, so refs are on
# the same scale as the training spectra (CRITICAL for reconstruction loss):
from src.data.preprocess import preprocess_batch
ref_tensor = preprocess_batch(ref_tensor)
np.save("engine/reference_spectra.npy", ref_tensor)
```

**This unblocks the reconstruction module (T13) in Phase A.**

### 2.8 `engine/bond_mapping.json` — ADD mineral entries (optional, Phase B)

For AAM dataset, we need bond entries for quartz and calcite. Paper says:
* **Quartz**: strong band at **464 cm⁻¹** (Si–O symmetric stretch)
* **Calcite**: strong band at **1058 cm⁻¹** (carbonate ν₁ symmetric stretch)
* Calcite secondary: **712 cm⁻¹** (in-plane bending), **1085 cm⁻¹**

Two new entries (add to `engine/bond_mapping.json`, **do not break existing 30**):

```json
{
  "id": "P031",
  "wavenumber_cm_inv": 464,
  "tolerance_cm_inv": 6,
  "bond": "Si-O symmetric stretch",
  "mode": "stretch",
  "compounds": ["Quartz"],
  "discriminative_for": ["Quartz"],
  "notes": "Diagnostic quartz peak in AAM dataset (Zarei 2023)."
},
{
  "id": "P032",
  "wavenumber_cm_inv": 1086,
  "tolerance_cm_inv": 6,
  "bond": "Carbonate symmetric stretch (ν1)",
  "mode": "stretch",
  "compounds": ["Calcite"],
  "discriminative_for": ["Calcite"],
  "notes": "Diagnostic calcite peak in AAM dataset; overlaps Glucosamine pyranose region (1080-1100)."
}
```

### 2.9 Folder structure additions

```
Raman-Physics-AI-v2/data/raw/
├── AA_Data.csv                       # primary (replaces data.csv)
├── AAM_Data.csv                      # AA + minerals (Phase B+)
├── AA_proportions.xlsx               # ground-truth ratios reference
├── AAM_proportions.xlsx
├── pure/                             # NEW subfolder — ENLIGHTEN raw exports
│   ├── DL-alanine.csv
│   ├── L-asparagine.csv
│   ├── L-aspartic-acid.csv
│   ├── L-glutamic-acid.csv
│   ├── L-histidine.csv
│   ├── D-glucosamine-HCl.csv
│   └── 50_50_quartz_calcite.csv      # mineral background
├── bacteria_ID/                      # NEW subfolder
│   ├── X_reference.npy               (60000, 1000)
│   ├── y_reference.npy
│   ├── X_finetune.npy                (3000, 1000)
│   ├── y_finetune.npy
│   ├── X_test.npy                    (3000, 1000)
│   ├── y_test.npy
│   ├── X_2018clinical.npy            (10000, 1000)
│   ├── y_2018clinical.npy
│   ├── X_2019clinical.npy            (2500, 1000)
│   ├── y_2019clinical.npy
│   └── wavenumbers.npy               (1000,)
├── API/                              # NEW subfolder
│   ├── API_data.csv                  (3511, 3278)
│   └── Product_information.xlsx      (33 compounds metadata)
├── ood_demo/                         # NEW subfolder
│   └── MoS2-160o-12h-ph5.txt         (single spectrum)
└── README.md                         # UPDATE to document all 5 datasets
```

The old `data.csv` can be deleted after confirming `AA_Data.csv` parses
correctly. To be safe, keep it as `data.csv.legacy` until Phase-A train
finishes.

---

## 3. Required changes — training strategy

### 3.1 Three-tier training plan

**Tier 1 — Primary (mandatory, all of Phase A):**
* Train and evaluate **only on AA dataset**.
* Use Scheme A (composition-OOD, 42/6/6 vials) as headline metric.
* Use Scheme A' (random 60/20/20) as upper-bound sanity check.
* This is exactly the T03 setup — *no change needed*.

**Tier 2 — Mineral extension (Phase B, if Day 7 mid-checkpoint passes):**
* Add AAM dataset to training. Two options:
  * **Option α (separate models):** train AA-only and AAM-only, compare.
  * **Option β (joint model):** extend output simplex from 6→8 (add quartz, calcite),
    pad AA labels with zeros for the mineral channels. Train on `AA ∪ AAM` (4378 + 12956 = 17334 spectra).
* Decision criterion: if AA-only model already meets `MAE ≤ 0.020` target
  comfortably (Day 7), pursue Option β. Otherwise stick with α.

**Tier 3 — External benchmarks (Day 13 stretch only):**
* Run frozen MVP on bacteria_ID and API datasets *as classification tasks*
  (not regression). The MVP's primary head won't apply directly; what we test
  is the **OOD detector** and the **peak extractor + symbolic mapper** —
  neither of which assumes the regression head.
* Specifically: feed bacteria_ID spectra → measure if our OOD score correctly
  flags them as OOD relative to the AA training distribution. Expected: yes,
  almost all of them, since they're entirely different chemistry.
* MoS2 spectrum → symbolic mapper should produce no matches (novel compound),
  novelty locator should report unmatched peaks.

### 3.2 The `mix_method` column unlocks a new ablation

`AA_Data.csv` has 4 mix methods:
* `Hand-Mixed_Amino_Acids` (1440 spectra)
* `Mechanically_Mixed_Amino_Acids` (1440 spectra)
* `Hand-Mixed_Amino_Acids_Perturbed` (1438 spectra)
* `Pure` (60 spectra, 6 vials)

This enables **a robustness test**: train on 1 mix method, test on the others.
The Mechanically-Mixed is the gold standard (most homogeneous); Hand-Mixed
introduces granular heterogeneity; Perturbed adds intentional noise. A model
that "understands" the spectrum should generalize across mix methods.

**Recommendation:** add `split_C_mix_method_ood` as Scheme C in
`src/data/splits.py` during Phase A. Hold out one mix_method as test set;
train on the others. Run last (after A and A' are working).

### 3.3 OOD evaluation gets two new test sets — for free

Previously, only Scheme B (component holdout) provided OOD evaluation, and
we noted it's broken on this data (every compound in 49/54 vials → train
pool too small). Now we have:

* **bacteria_ID test set** (3000 spectra, 30 species) — *cross-domain* OOD.
  Expected behavior of MVP: every spectrum flagged OOD, predicted composition
  meaningless.
* **MoS2 spectrum** — *single-sample* OOD. Symbolic mapper detects no matches.
* **Mineral peaks in AAM** at 464 (quartz) and 1086 (calcite) — peaks that
  don't appear in pure AA training data, so the novelty locator should flag
  them even when AA composition is also predicted.

**These give the OOD AUROC metric something to measure** — previously we
would have had to synthesize OOD samples. Now we have real ones at three
difficulty levels:
* Easy: bacteria_ID (completely different chemistry, different instrument)
* Medium: MoS2 (different chemistry, similar instrument range)
* Hard: AAM (overlapping organic peaks, mineral interference)

### 3.4 Reconstruction loss now has *real* references

T13 (reconstruction module) was specified as `s_recon = Σ α_i · scale_i · pure_i`
where `pure_i` came from "mean of pure samples in training data". With the
ENLIGHTEN exports we now have **independently measured pure spectra** — same
instrument, same conditions, but separate measurements from the mixture data.
This is a much cleaner reference than averaging 10 pure samples that
participated in training.

**Decision:** Use ENLIGHTEN exports as canonical references (build via
`scripts/extract_pure_references.py`). Cross-validate against
"averaged-from-training" references to confirm they match within tolerance.

---

## 4. What does NOT need to change

To prevent over-correction, here's what stays exactly as-is:

* **All four config files structure** — only the `paths` / `datasets` section
  changes; all other sections (model, training, OOD) are unaffected.
* **`src/data/splits.py`** — Schemes A and A' work unchanged. Scheme B
  remains documented-but-deprecated. Scheme C is *added* (not modifying
  existing code).
* **`src/data/preprocess.py`** — pipeline is dataset-agnostic; works on any
  (N, P) numpy array of spectra. The `is_preprocessed` flag handles
  bacteria_ID's pre-normalized data.
* **`engine/symbolic_mapper.py`** — class is JSON-driven; just needs the
  JSON updated to include mineral entries.
* **`tests/test_data.py` and `tests/test_engine.py`** — existing tests
  remain valid; new tests will be *added* for the new loaders, not replacing
  old ones.
* **All Phase-A architectural decisions** (1D-ResNet backbone, softmax
  simplex, MC Dropout 50, physics reconstruction loss) — unchanged.

---

## 5. Updated 14-day timeline

| Phase | Days | Tasks | What's new vs original plan |
|---|---|---|---|
| Groundwork | 1–2 | T01–T05 | DONE (this revision documents follow-up cleanup) |
| **Phase A (Modeling)** | 3–7 | T11–T17 + new dataloader refactor | **+2 hours** for dataloader registry refactor (Section 2.3) and `extract_pure_references.py` (Section 2.7). Cut from somewhere (recommend: skip wandb logging setup for now). |
| Mid-checkpoint | 7 | GO/NO-GO on AA-only MAE ≤ 0.040 | unchanged |
| Phase B (Inference) | 8–10 | T18–T22 + AAM extension | If GO: train AAM-extended model (Tier 2). Otherwise stay on AA. |
| Phase C (Eval) | 11–12 | T24–T26 + Scheme C | **+ Scheme C (mix_method OOD)** |
| Phase D (Stretch) | 13 | T28 + external benchmarks | **NEW**: bacteria_ID OOD test, MoS2 demo, API classification (only if Day 12 ahead) |
| Final | 14 | report, tests, tag | unchanged |

**Net cost:** ~2 extra hours of dataloader work in Phase A, fully repaid
by getting `engine/reference_spectra.npy` built from real data and getting
3 free OOD test cases.

**Net benefit:** the thesis now has:
* A reproducible link to a published paper (Zarei 2023)
* External benchmarks to claim generalization
* A concrete OOD demonstration (MoS2 → "novel material detected")
* Mineral-extended dataset for v2 future work

---

## 6. Concrete file checklist for next chat (Phase A)

Before starting modeling work, the next chat should:

1. ☐ Create `data/raw/{pure,bacteria_ID,API,ood_demo}/` subfolders.
2. ☐ Drop the new files into the right places per Section 2.9.
3. ☐ Move old `data.csv` → `data.csv.legacy` (or delete after AA verification).
4. ☐ Update `configs/default.yaml` per Section 2.1 (dataset registry).
5. ☐ Refactor `src/data/dataloader.py` per Section 2.3 (format dispatch).
6. ☐ Add `src/data/enlighten_parser.py` per Section 2.4.
7. ☐ Add `src/data/bacteria_id_loader.py` per Section 2.5.
8. ☐ Implement `scripts/extract_pure_references.py` per Section 2.7.
9. ☐ Update `engine/bond_mapping.json` with quartz + calcite entries (Section 2.8).
10. ☐ Re-run `pytest tests/test_data.py tests/test_engine.py` — must still pass.
11. ☐ THEN start T11 (1D-ResNet backbone).

Items 1-10 are estimated **2-3 hours total**. Item 11 onwards is the actual
Phase A modeling work.

---

## 7. Open questions for the user

1. **Storage budget on Windows.** With AA + AAM + reference + bacteria_ID +
   API, raw data is **~960 MB**. Is `E:\Project\KhoaLuanCourse\` OK with that?
   If concerned, we can keep AAM and bacteria_ID outside the project tree and
   reference them by absolute path in the registry.

2. **AAF dataset.** The Zarei paper has a third dataset, AAF (amino acids +
   fluorescence), 2880 spectra. We don't have it — is it available? It would
   complete the trilogy and let us claim "validated on all three Zarei datasets".

3. **bacteria_ID classification head.** Phase A trains a 6-way regression
   head (AA composition). bacteria_ID is 30-way classification. Three options:
   (a) freeze the MVP and only use bacteria_ID for OOD score testing,
   (b) add a separate classification head trained on bacteria_ID,
   (c) defer entirely to v2.
   **Recommendation: (a)** — minimal scope creep, still gets us the OOD
   evaluation evidence. Confirm before Phase C.

4. **Class label semantics for bacteria_ID and API.** We need the species/API
   names corresponding to integer class IDs. For bacteria_ID, the Ho 2019
   paper supplement provides them; for API, `Product_information.xlsx` looks
   like the right reference but I'd like you to confirm which sheet/column
   maps integer ID → name.

---

*Document version 1.0 — written immediately after Groundwork phase, in
preparation for Phase A. Read this together with the updated
`docs/GROUNDWORK_SUMMARY.md` (next file) which incorporates these changes
into the standing handover.*
