# Chat 2 Phase A — Handover Summary

> **Use this document at the start of Chat 3 (Core Engine) and any return
> trips to Chat 2 (T17 mid-checkpoint, Phase B benchmarks).** It records
> what Phase A produced, what's still broken, and which decisions diverged
> from the original plan.
>
> **Status:** T06 + T07 + T09 done end-of-Phase-A. T17 (mid-checkpoint),
> T23/T24 (baselines), T25 (comparison) still pending — they need a
> trained model from Chat 3 first.

---

## A. Files created in Phase A

All 5 files drop into the existing project tree at `E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\`.

| Path | Purpose | Task | Lines |
|---|---|---|---|
| `src/data/enlighten_parser.py` | Parse Wasatch ENLIGHTEN proprietary CSV (~30 metadata rows + interleaved Processed columns). Returns `(N, P)` spectra + wavenumber axis. | T06 helper | ~220 |
| `scripts/extract_pure_references.py` | CLI: read 6 ENLIGHTEN pure CSVs → aggregate (mean / median fallback) → resample to AA wavenumber grid → optional T04 preprocessing → save `engine/reference_spectra.npy`. Also writes per-compound `.npy`, manifest JSON, and `results/sanity/pure_spectra.png`. | T06 main | ~360 |
| `src/data/splits.py` | **REPLACES** Groundwork T03 version. Functions: `split_A_vial_level()` (42/6/6 vials), `split_A_sample_level()` (60/20/20 random), `is_pure_vial()`, `save_split()`, `load_split()`. JSON format: `{"train":[...], "val":[...], "test":[...], "scheme":..., "seed":...}`. | T07 | ~285 |
| `src/eval/metrics.py` | 5 metrics: `quantification_mae`, `identification_accuracy`, `ood_auroc` (sklearn + Mann-Whitney fallback), `constraint_violation_rate`, `reconstruction_cosine_similarity`. Each with formula docstring + edge cases. | T09 | ~290 |
| `tests/test_metrics.py` | 23 pytest unit tests with hand-computed expected values. **All 23 PASS** in sandbox. | T09 | ~210 |

**Files NOT created in Phase A** (deferred / out of scope):
- `engine/reference_spectra.npy` — built BY `extract_pure_references.py` when user runs it on real data
- `data/splits/split_A_composition_ood.json` — built when user runs `split_A_vial_level()` on real cache
- All Phase B deliverables (mid-checkpoint, baselines, comparison)

---

## B. Open issues / known constraints

### Issue #1 — Splits.py REPLACES Groundwork T03 version, doesn't merge
The new `splits.py` has a different API from the T03 file (deliberately, per T07 spec).
- **Removed:** Scheme B (component-OOD) and Scheme C (mix_method) — those were
  documented as deprecated/deferred in PROJECT_REVISION §3.2.
- **Removed:** `split.scheme: "all"` runner.
- **Action:** if any other Groundwork code imports from old `splits.py` (e.g.
  `from src.data.splits import split_C_mix_method_ood`), those imports will fail.
  Search-and-replace or restore from `splits.py.bak_t03`.

### Issue #2 — `extract_pure_references.py` requires `src.data.preprocess.preprocess_batch`
Script imports `preprocess_batch(np.ndarray (N,P)) → np.ndarray (N,P)` from T04.
If signature in actual T04 file differs, the `--apply-preprocessing` flag falls
back to identity (with stderr warning). **Verify T04 exposes that exact name** —
if not, either rename in T04 or change the import in `extract_pure_references.py`.

### Issue #3 — Reference vs cached spectra alignment unverified
`extract_pure_references.py` resamples ENLIGHTEN refs onto `data/processed/wavenumbers.npy`
(AA's grid). This assumes both grids overlap reasonably. If AA covers ~267-2004
cm⁻¹ but ENLIGHTEN covers ~267-2004 cm⁻¹ (per Zarei 2023 spec), they should match.
**Action:** after running step 6, plot `pure_spectra.png` and visually verify
Histidine peaks land at ~1003/1180/1495/1575 cm⁻¹.

### Issue #4 — Multi-dataset refactor (PROJECT_REVISION §6) NOT done by Phase A
Chat 2 Phase A only built T06/T07/T09. The dataloader-registry refactor
(`load_dataset(name, registry)`, ENLIGHTEN dispatcher, bacteria_ID loader,
`split_C_mix_method_ood`) is **still pending** and is Chat-1-leftover work.
**Recommendation:** do this BEFORE Chat 3 starts T11, OR defer until Phase B
when AAM gets pulled in.

### Issue #5 — Compound-coverage assertion in `split_A_vial_level` may fail
The function raises if any compound has 0 train rows. With `include_pure_in_train=True`
this should never happen (every compound has a pure vial in train). With
`include_pure_in_train=False` and bad luck, a Histidine-only training set could
miss e.g. Glucosamine → ValueError. **Mitigation:** keep default True.

### Issue #6 — Subsampling for large datasets DEFERRED
User decided AAM 270→30 spectra/vial (9× reduction) and similar policies for
bacteria_ID `X_reference.npy` (60K), `X_2018clinical.npy` (10K), API. None of
this affects Phase A. **Action:** implement in Phase B / Phase D loaders, not
in current `extract_pure_references.py`.

### Issue #7 (carried over from Groundwork) — Scheme B essentially broken
Every compound appears in 49/54 vials → component-OOD train pool ≈ 50 rows.
Phase A removed Scheme B from `splits.py` entirely. If thesis needs to claim
"we evaluated component OOD too", document Scheme B as stress-test-only with
this caveat.

---

## C. Decisions changed vs. original plan

| # | Original plan (Custom Instructions / Chat 2 message) | What Phase A actually did | Reason |
|---|---|---|---|
| C1 | Compound order: `Alanine, Glycine, Serine, Threonine, Histidine, Glucosamine` (T06 spec in message) | **Used canonical order from Custom Instructions §3 + GROUNDWORK_SUMMARY: `Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine`** | T06 message spec was stale (pre-revision). New order matches `AA_Data.csv` label cols and `engine/reference_spectra.npy` rows. |
| C2 | T06 builds refs by filtering pure vials from `data.csv` | **Built from 6 separate ENLIGHTEN exports** (PROJECT_REVISION §2.7) | Cleaner refs (no training-data leak); same instrument = compatible scale. |
| C3 | Scheme A vial split: 42/6/6 random over all 54 vials | **Force 6 pure vials into train; split 42/6/6 with 36 train mixtures + 6 val mixtures + 6 test mixtures** (`include_pure_in_train=True` default) | Reconstruction module needs pure-mixture pairs in same split; OOD eval should test composition novelty, not compound novelty. |
| C4 | Expected split sizes ~3538/460/380 (per Groundwork doc) | **3298/540/540** | Different vial counts per group + different per-vial spectrum counts. Old number assumed different `include_pure` policy. |
| C5 | Reference preprocessing automatic | **Optional via `--apply-preprocessing` flag** | T04 may not be import-stable in all environments; explicit flag forces user to think about scale alignment. |
| C6 | Splits.py keeps Schemes B + C | **Removed B and C from Phase A** (only A and A' implemented) | T07 spec lists only A and A'; B is broken on this data; C was a Phase B/C addition. Add back when needed. |
| C7 | Tests use pytest natively | **Uses pytest API, but tested in sandbox via custom shim** (env had no pytest) | Tests still work with `pytest` on user's machine; the shim was sandbox-only. |
| C8 | OOD AUROC requires sklearn | **Falls back to manual Mann-Whitney implementation if sklearn absent** | Robustness. |
| C9 | Reconstruction cosine returns scalar | **Returns dict (median + mean + 5/25/75/95 percentiles)** | Distribution can be bimodal; single number hides that. CVR consumes the same dict. |

---

## D. What Chat 3 (Core Engine) needs from Phase A

When Chat 3 starts, these are the contracts it should depend on:

1. **`engine/reference_spectra.npy`** — `(6, 1024)` float32, rows in canonical compound
   order (Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine).
   Reconstruction module (T13) must use the same ordering.

2. **`data/splits/split_A_composition_ood.json`** — `{"train":[...], "val":[...], "test":[...], "scheme":"A_vial_level", "seed":42}`. Train loader reads `train` list; val loader reads `val`; test only used at T17 / T25.

3. **`src.eval.metrics`** — call from training loop validation:
   ```python
   from src.eval.metrics import quantification_mae, reconstruction_cosine_similarity
   val_mae = quantification_mae(y_true, y_pred)
   recon_stats = reconstruction_cosine_similarity(s_in, s_recon)
   logger.log({'val/mae': val_mae, 'val/recon_median': recon_stats['median']})
   ```

4. **Compound order is locked** — do NOT reorder in models, configs, or `bond_mapping.json`
   without coordinating with `engine/reference_spectra.npy`.

---

## E. Return-to-Chat-2 trigger conditions

After Chat 3 reports "model trained, checkpoint saved at `checkpoints/best.pt`,
training_log.csv complete", come back to Chat 2 to run:

- **T17 mid-checkpoint** — load `best.pt`, evaluate on val set with all 5 metrics
  from `src/eval/metrics.py`, write `results/midcheckpoint_report.md`, GO/NO-GO
  decision.

After T17 GO, Chat 3 continues to Phase 3 (inference module). After Phase 3, return
to Chat 2 once more for T23 (PCA+SVM baseline), T24 (ResNet-only baseline), T25
(comparison table). All Phase B work uses the same `metrics.py` from Phase A.

---

## F. Quick reproduction recipe (Windows PowerShell)

```powershell
# Prerequisites: T01-T05 done, multi-dataset refactor done (PROJECT_REVISION §6)

# 1. Drop 5 Phase A files into project (overwrite splits.py)
# 2. Verify metrics tests
pytest tests\test_metrics.py -v       # expect 23 passed

# 3. Rebuild AA cache (if not done)
python scripts\prepare_data.py --dataset AA --scheme all

# 4. Build splits
python -c "
import numpy as np, torch
from src.data.splits import split_A_vial_level, split_A_sample_level, save_split
labels   = torch.load('data/processed/labels.pt').numpy()
vial_ids = np.load('data/processed/vial_ids.npy', allow_pickle=True).tolist()
save_split(split_A_vial_level(vial_ids, labels, seed=42),
           'data/splits/split_A_composition_ood.json')
save_split(split_A_sample_level(len(vial_ids), labels, seed=42),
           'data/splits/split_A_prime_sample_level.json')
"

# 5. Build reference spectra
python scripts\extract_pure_references.py `
    --pure-dir data\raw\pure `
    --target-wavenumbers data\processed\wavenumbers.npy `
    --apply-preprocessing

# 6. Visual sanity check
start results\sanity\pure_spectra.png
```

After step 6 returns clean: **Phase A complete. Move to Chat 3.**

---

*End of Chat 2 Phase A handover. Document version 1.0.*
