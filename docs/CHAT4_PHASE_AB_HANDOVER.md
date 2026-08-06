# Chat 4 Phase A-B (T20 + T21 + T22) — Handover Summary (v1.1)

> **Use this document at the start of:**
> 1. **Chat 2 Phase B** (T23 PCA+SVM, T24 ResNet-only, T25 comparison) — **NEXT**.
> 2. **Chat 4 Phase C-D** (T26 report, T27 visualisation, T_glue
>    `predict()`, optional T28 Streamlit dashboard).
> 3. **Chat 5 integration** at the end.
>
> **Also attach:** `CHAT3_PHASE3_HANDOVER.md` (T18+T19 inference primitives),
> `CHAT3_CORE_ENGINE_HANDOVER.md` (model checkpoint contract),
> `CHAT2_TASK17_HANDOVER.md` (test-set verdict, GO-WITH-CAVEAT).
>
> **Status:** T20 (peak extraction) + T21 (symbolic mapper) + T22 (novelty
> locator) DONE and **deployed locally**. **281/281 tests pass** on user's
> Windows machine (`E:\Project\KhoaLuanCourse\`, Python 3.13.7, pytest 9.0.3,
> ~40 s wall time). Real-data verification ran on a real His-dominant mixture
> from `data.csv`: Histidine identified correctly, but disambiguation surfaced
> known limitations (see §B.2 P4AB-3 update and §B.4 new finding).
>
> **What changed in v1.1 (vs v1.0 handover):** the user ran the modules on
> real `data/processed/*.pt` cache and discovered that DB tolerances tuned
> for synthetic data were too tight for real Wasatch WP-785 spectra.
> Tolerances of 6 entries (P004, P005, P006, P008, P013, P014) were widened
> from 4-6 cm-1 to **8 cm-1** to accommodate calibration drift. Tests in
> `tests/test_symbolic_mapper.py` were updated to use the new tolerances.
> All 72 engine tests still pass; 281 total tests across the project pass.

---

## A. Files created in Phase A-B (final state on disk)

All paths relative to `E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\`.

### A.1 Source modules (production)

| Path | Module | Lines | Purpose |
|---|---|---|---|
| `engine/peak_extractor.py` | T20 | 455 | `PeakExtractor` class + `Peak` dataclass. Wavenumber-aware peak detection: `find_peaks_basic()` (scipy.signal) -> `fit_voigt()` (lmfit.VoigtModel) -> `extract_full()`. Handles both ascending and descending wavenumber axes (AA data uses descending). Sub-pixel position refinement, R^2 fit quality, FWHM in cm-1. Includes `pixel_to_cm()` / `cm_to_pixel()` helpers and a Gaussian-FWHM fallback when lmfit fits fail. |
| `engine/symbolic_mapper.py` | T21 (+T05 base) | 460 | `BondMapper` class (loads `bond_mapping.json`) + `BondEntry` / `AnnotatedPeak` dataclasses. **Base layer (T05)**: `match_peak(wavenumber, tolerance_override)`. **T21 enhancements**: `compute_match_confidence(peak, entry)` -> `{high, medium, low}`, `annotate_peaks(peaks, prefer_discriminative)` -> `list[AnnotatedPeak]`, `disambiguate_compound(annotated)` -> `{votes, likely_compounds, unsupported_compounds}` using weighted-vote scheme (high=1.0, medium=0.5). |
| `engine/novelty_locator.py` | T22 | 343 | `NoveltyLocator` class + `PeakCluster` dataclass. Single-link clustering of unmatched peaks by position proximity (default gap 30 cm-1). Suggests chemistry via a coarse cm-1-region lookup table (14 regions covering 100-3700 cm-1, including MoS2 lattice modes ~380/408, amide I ~1500-1700, O-H/N-H ~3200-3700). Pipeline: `find_unmatched_peaks` -> `cluster_unmatched` -> `suggest_chemistry`, plus end-to-end `locate()`. |
| `engine/bond_mapping.json` | T05 seed + T21 use + **v1.1 retune** | 318 | **30-entry seed DB** built from De Gelder 2007 + Socrates 3rd ed. Schema: `{id, wavenumber_cm_inv, tolerance_cm_inv, bond, mode, compounds, discriminative_for, notes}`. **v1.1 tolerance retune:** P004 (Histidine 1003) 4->8, P005 (Glucosamine 1080) 6->8, P006 (Glucosamine 1100) 6->8, P008 (Histidine 1180) 5->8, P013 (Histidine 1495) 6->8, P014 (Histidine 1575) 6->8. All retuned entries' `notes` field documents the change. Discriminative coverage unchanged. |
| `engine/__init__.py` | new | 8 | Package docstring describing the four siblings. |

### A.2 Tests

| Path | Tests | Coverage |
|---|---|---|
| `tests/test_peak_extractor.py` | **21 tests** in 4 classes (`TestConstruction`, `TestFindPeaksBasic`, `TestFitVoigt`, `TestExtractFull`) | ascending+descending axis, pixel<->cm round-trip, non-monotonic rejection, 4-imidazole Histidine recovery, sub-pixel Voigt refinement (R^2 > 0.95), FWHM in [2, 30] cm-1 range, noise rejection by relative-height threshold, post-filter merging of duplicate detections, orientation invariance. |
| `tests/test_symbolic_mapper.py` | **27 tests** in 6 classes — **v1.1 retuned** | Same coverage as v1.0; confidence-boundary tests updated to use the new tol=8 windows on P004. Key updated cases: `test_high_confidence_at_center` checks `|delta|=1 < 4`, `test_medium_confidence_near_edge` uses `|delta|=4.5`, `test_low_confidence_outside` uses `|delta|=9`, `test_medium_confidence_half_weight` uses `|delta|=6` (within tol but >= tol/2). |
| `tests/test_novelty_locator.py` | **24 tests** in 6 classes (`TestFindUnmatched`, `TestCluster`, `TestRegionHints`, `TestLocate`, `TestCustomGap`) | Unmatched filter from raw dicts and from `AnnotatedPeak`, single-link chain clustering, intensity-weighted centroid, MoS2 phonon labelling, amide-I labelling, O-H/N-H labelling, end-to-end `locate()` on MoS2 / pure-Histidine / Histidine+novel-peak, JSON-serialisable output, narrow-gap-splits / wide-gap-merges. |
| `tests/__init__.py` | new | 0 | empty marker. |

**Final test status on user's Windows machine** (`pytest tests\ -v --tb=short`):

```
collected 281 items
...
====================== 281 passed in 39.61s ======================
```

Broken down: 20 augmentation + 39 data + 25 engine (T05 base) + 28 metrics + 31 models + 24 novelty_locator + 34 ood + 21 peak_extractor + 27 symbolic_mapper + 32 uncertainty = 281. **All previous Chat 1/2/3 tests remain green** — no regression introduced by Phase A-B.

### A.3 Sanity demos

| Path | Purpose | Status |
|---|---|---|
| `results/sanity/peak_demo_histidine.png` | Annotated plot: synthetic Histidine spectrum (4 imidazole peaks at 1003/1180/1495/1575 cm-1, Gaussian sigma=4-6 cm-1, sigma=0.01 noise floor) with Voigt-refined peak markers + R^2/FWHM annotations. **All 4 peaks recovered at sub-cm-1 precision, R^2 > 0.99, FWHM 9-14 cm-1.** | Committed |
| `results/sanity/peak_demo_histidine_real.png` | (Per the local guide, P4AB-1 mitigation) Annotated plot from a real pure-Histidine row in `data/processed/spectra_full.pt`. | Generated locally during verification, see scripts/real_histidine_demo.py |
| `scripts/real_histidine_demo.py` | One-off verification script: load cache -> find pure-His row -> extract peaks -> annotate -> locate novelty -> save plot. | Committed |
| `tests/test_mixture.py` (user-authored ad hoc) | Loads mixture row from cache and runs the full extract -> annotate -> disambiguate pipeline. Used during retune to compare DB tolerances vs real-data peak positions. | NOT a pytest test; one-off script. Consider moving to `scripts/` in cleanup. |

### A.4 Files NOT created in Phase A-B (downstream scope)

| Path | Task | Owner |
|---|---|---|
| `src/inference/predict.py` | T_glue -- main `predict(spectrum)` orchestrating model+T18+T19+T20+T21+T22 | **Chat 4 Phase C** |
| `src/inference/report.py` | T26 -- JSON + Markdown per-sample report generator | **Chat 4 Phase C** |
| `src/inference/visualize.py` | T27 -- reconstruction overlay / peak annotations / OOD summary plots | **Chat 4 Phase C** |
| `results/reports/demo_*.md` x 3 | Demo reports (1 ID, 1 mild OOD, 1 fully novel) for thesis defense | **Chat 4 Phase C** |
| `src/models/baselines/pca_svm.py` | T23 PCA + SVM baseline | **Chat 2 Phase B (next)** |
| `src/models/baselines/resnet_only.py` | T24 ResNet-only (no physics) baseline | **Chat 2 Phase B (next)** |
| `src/eval/benchmark.py`, `results/benchmark_table.md` | T25 comparison table | **Chat 2 Phase B (next)** |
| `dashboard/app.py` | T28 -- Streamlit upload-and-analyse UI | **Chat 4 Phase D** (optional) |
| `src/data/bacteria_id_loader.py`, `src/data/mos2_loader.py` | external benchmark loaders | **Chat 4 Phase D** (optional) |

---

## B. Open issues / known constraints

### B.1 Carried over from earlier phases (still relevant)

| # | Issue | Origin | Phase A-B impact | Downstream impact |
|---|---|---|---|---|
| T17-B1 | Cache built from legacy `data.csv` (vial naming `a01-a48`) | Chat 2 T17 | None -- engine modules are dataset-agnostic | T23/T24 baselines and T_glue `predict()` will read same cache. Document. |
| T17-B2 | `engine/reference_spectra.npy` not built from ENLIGHTEN; baked into checkpoint | Chat 2 T17 | None -- engine modules don't use refs directly | T_glue should NOT rebuild refs; load from checkpoint's `reconstruction.pure_ref` buffer (per CHAT3_PHASE3_HANDOVER D.2) |
| T17-B6 | MAE ceiling ~0.055 -- fails Day-7 gate | Chat 2 T17 | None | T26 report **must** surface composition_std alongside means so the high-uncertainty story (Glucosamine, Asparagine) is honest. T23/T24 baselines should be evaluated with the **same** test split for fair comparison. |
| P3-1 | Sandbox had no torch -- Phase 3 PyTorch idioms unverified | Chat 3 Phase 3 | Confirmed resolved -- 32 uncertainty + 34 ood tests now pass on real hardware | None for downstream |
| P3-2 | `OODScorer.calibrate()` cost = `n_val * (1 + mc_samples)` forwards | Chat 3 Phase 3 | None | Phase C should reuse `calibration.json` from the stretch run rather than re-calibrating |
| P3-3 | `OODScorer` must be fed the raw model, not `MCDropoutWrapper(model)` | Chat 3 Phase 3 | None | T_glue: build `model.eval()`, pass directly to `OODScorer.from_file(model, "calibration.json")` |

### B.2 New issues from Phase A-B (v1.0)

| # | Issue | Severity | Status / Mitigation |
|---|---|---|---|
| **P4AB-1** | Sanity demo originally synthetic-only | Medium | **RESOLVED in local verification.** User generated `results/sanity/peak_demo_histidine_real.png` via `scripts/real_histidine_demo.py`. Real data shows calibration drift (see B.4 below); led to v1.1 DB retune. |
| **P4AB-2** | `BondMapper.match_peak` can return multiple entries when peak windows overlap; `annotate_peaks` resolves with `prefer_discriminative=True`. After v1.1 retune (more windows at tol=8), overlaps increase slightly. | Low | Still resolved by `prefer_discriminative=True` default. Tests confirm P004 wins at 1003 cm-1 even when P003 (1000±8) now also matches. |
| **P4AB-3** | `disambiguate_compound` single-vote threshold can produce false positives on noisy spectra. | **Medium -- CONFIRMED on real data, see B.4.** | Phase C should implement the cross-check pattern (§D.3) -- compare learned-head composition vs symbolic-head likely_compounds and flag disagreement. |
| **P4AB-4** | `NoveltyLocator` region table hard-coded. | Negligible | `region_table=...` ctor arg available for v2. |
| **P4AB-5** | `Peak` and `AnnotatedPeak` are separate dataclasses (deliberate decoupling). | Low | T26 should accept `AnnotatedPeak` only. |
| **P4AB-6** | Mineral entries P031 (quartz 464) and P032 (calcite 1086) NOT in seed DB. | Negligible for MVP | Wait for AAM extension. |
| **P4AB-7** | Voigt fit fallback silently drops peaks below `fit_quality_threshold=0.8`. | Low | Phase C could emit a stderr note when `len(coarse) > 0 and len(kept) == 0`. |

### B.3 New issues NOT in v1.0 but discovered during local verification

| # | Issue | Severity | Mitigation |
|---|---|---|---|
| **P4AB-8** | **Bond DB tolerances tuned for synthetic data were too tight.** When run on real `data.csv` rows, the 4 Histidine imidazole peaks and 2 Glucosamine pyranose peaks frequently landed 4-6 cm-1 off their nominal positions due to Wasatch WP-785 calibration. Original tol=4-6 entries gave `confidence=low` or `matched_to=None`, breaking disambiguation on pure-Histidine samples. | **Resolved in v1.1**: tolerances widened to 8 cm-1 for P004, P005, P006, P008, P013, P014. Each entry's `notes` field documents the change. Test file updated to use new tol=8 in boundary cases. | **No further action.** But Phase B/C should be aware that the 8 cm-1 figure was tuned to *this* spectrometer and that papers reporting different instruments may want different tolerances. |
| **P4AB-9** | **Mixture disambiguation has both false positives and false negatives** even after the v1.1 retune. Real-data smoke test on row 780 (gt: His 0.45, Glc 0.19, Ala 0.15, Asn 0.10, Glu 0.08, Asp 0.03): `likely_compounds = ['Glutamic Acid', 'Histidine']`. Histidine correctly identified. But: (a) Glutamic Acid (gt 0.08, smallest non-Asp component) gets a vote — false positive; (b) Glucosamine (gt 0.19) is absent — false negative. This is the issue P4AB-3 predicted, now observed. | **Medium** for the thesis story. The symbolic head is meant as an *interpretability* layer, not a competing quantification head, but reviewers will ask. | Phase C T26 should: (1) ALWAYS show both learned-head composition AND symbolic-head likely_compounds, never just one; (2) document the cross-check disagreement explicitly per the §D.3 pattern; (3) frame the symbolic head as "hypotheses with confidence" rather than "answers". Phase C may also consider raising `min_discriminative_hits` from 1.0 to 1.5 — would suppress the Glutamic-Acid FP (only one P020 discriminative peak hit with weight 1.0 = exactly threshold) without losing Histidine (4 peaks). |
| **P4AB-10** | An ad-hoc `tests/test_mixture.py` was added by the user during verification but is NOT a pytest test — it's an executable script that prints to stdout. Pytest may collect it on next run, but it doesn't follow the test naming/assertion convention. | Cosmetic | Move to `scripts/test_mixture_smoke.py` in a cleanup pass; or convert to a real pytest fixture-based test. |

### B.4 Real-data observations worth flagging to thesis defense

Three things the user observed during local verification that should be in the thesis narrative:

1. **Calibration drift on real Wasatch WP-785 is ~3-5 cm-1.** The 8-cm-1 tolerance in the v1.1 DB is empirically justified by the gap between expected (literature) and observed peak positions on real `data.csv` rows. This is honest physics — not a model failure.

2. **Pure-compound disambiguation works.** On pure-Histidine samples, `disambiguate_compound` returns `["Histidine"]` cleanly. The symbolic layer recovers the right answer when the chemistry is unambiguous.

3. **Mixture disambiguation is noisy at low-mixing-ratio components.** The Glutamic-Acid false positive at row 780 (gt 0.08) and Glucosamine false negative (gt 0.19) tell the same story: a single discriminative peak hit is enough to flag a compound regardless of its actual amount. This is **by design** — the symbolic head answers "is X plausibly present?" not "how much X is there?" — but the report (T26) must frame it that way to avoid being misleading.

### B.5 NOT done in Phase A-B (out of scope by design)

- **Mineral entries** (P4AB-6): wait for AAM extension.
- **PeakExtractor preprocessing integration**: the extractor assumes the input is already preprocessed (AsLS baseline + cosmic + SG + SNV per Custom Instructions 2). T_glue is responsible for routing raw -> preprocessed before calling the extractor.
- **OOD peak attribution**: when a sample is flagged OOD by T19, the novelty locator's clusters are the *interpretation*. T19's score remains the canonical OOD signal. The locator is descriptive, not detective.

---

## C. Decisions that diverged from the original plan

### C.1 Spec-level decisions (planned divergences vs the chat-task message)

Same as v1.0 handover (D1 -- D9). See archived v1.0 if needed; abbreviated here:

- **D1** Synthetic Histidine demo (real-data demo added later, see B.2 / P4AB-1).
- **D2** `find_peaks_basic` returns position in cm-1, not pixel index.
- **D3** `fit_voigt` returns lmfit's `height` (actual peak height) as `intensity`, not integrated amplitude.
- **D4** `fit_quality_threshold` is a ctor arg, not hardcoded.
- **D5** T21 returns `AnnotatedPeak` dataclass, not dicts.
- **D6** `compute_match_confidence` omits the "intensity match expected" clause (DB has no expected-intensity field).
- **D7** `disambiguate_compound` uses weighted vote (high=1.0, medium=0.5) over `discriminative_for`.
- **D8** `cluster_unmatched` uses single-link with default gap 30 cm-1.
- **D9** `suggest_chemistry` returns human-readable strings + structured `region_hints`/`region_examples` on each `PeakCluster`.

### C.2 v1.1 runtime decisions (NEW in this version)

Made by user during local deployment after observing real-data behaviour:

| # | What changed | Reason | Result |
|---|---|---|---|
| **R1** | Tolerance widened from 4 cm-1 to **8 cm-1** for entry P004 (Histidine 1003 imidazole breathing) | On real pure-Histidine rows from `data.csv`, the 1003 cm-1 imidazole breathing peak landed 4-6 cm-1 off the literature value due to instrument calibration. With tol=4, the peak landed in the medium-or-low confidence band, breaking unanimous Histidine voting. | Pure-Histidine samples now get `confidence=high` matches and `likely_compounds=["Histidine"]`. |
| **R2** | Tolerances widened for P005 (1080), P006 (1100), P008 (1180), P013 (1495), P014 (1575) -- all 6 to 8 | Same Wasatch calibration drift affects all imidazole and pyranose peaks roughly equally. Widening all 6 uniformly is the conservative move. | Confirmed -- Glucosamine pyranose peaks now match high-confidence on pure-Glc rows; same for Histidine. |
| **R3** | `tests/test_symbolic_mapper.py` boundary cases updated to use the new tol=8 windows on P004 | The original tests used tol=4 boundaries (`|delta|=2`, `|delta|=3`, etc); after R1+R2, those values are now well inside `tol/2=4` and would all flip to "high", invalidating the test. | All 27 symbolic_mapper tests still pass with the new boundary values: `|delta|=1 -> high`, `|delta|=4.5 -> medium`, `|delta|=9 -> low`. |
| **R4** | Each retuned entry's `notes` field documents the change in-place, with a fixed string: "Tolerance widened from X to Y to accommodate instrument calibration drift (Zarei 2023 / Wasatch WP-785) observed in real mixture spectra." | Forensic traceability -- future maintainers (or v2 datasets) can see the change was empirical and what instrument it was tuned for. | Documented. |
| **R5** | User added `scripts/real_histidine_demo.py` and `tests/test_mixture.py` (the latter is an ad-hoc smoke script, not a pytest test) | Verification of the engine modules against real cache data. The mixture script printed the row-780 disambiguation that exposed P4AB-9. | Both files committed locally. `test_mixture.py` could be moved to `scripts/` (see P4AB-10). |

**None of these changes touched Python source code in `engine/`.** All retunes were data-only edits to `bond_mapping.json` + corresponding test-side fixture updates. The module APIs and contracts are unchanged from v1.0.

---

## D. Contracts downstream chats will rely on

### D.1 Compound order LOCKED (carried from Chat 3)

```python
COMPOUND_ORDER = ["Alanine", "Asparagine", "Aspartic Acid",
                  "Glutamic Acid", "Histidine", "Glucosamine"]
```

Matches: model output, `engine/reference_spectra.npy` rows, `bond_mapping.json` `compounds` field, `bond_mapping.json` metadata `compound_order` field. **Do not reorder.**

### D.2 API contract for T_glue `predict()` (Phase C)

```python
# Imports T20 + T21 + T22 will need:
from engine.peak_extractor   import PeakExtractor
from engine.symbolic_mapper  import BondMapper, AnnotatedPeak
from engine.novelty_locator  import NoveltyLocator

# Construction (once, at predict() module load -- these are stateless given inputs):
wavenumbers = np.load("data/processed/wavenumbers.npy")        # (1024,) float64
extractor = PeakExtractor(wavenumbers,
                          height=0.05, prominence=0.03, distance=10,
                          fit_window_cm=30.0, fit_quality_threshold=0.8)
mapper = BondMapper.from_json("engine/bond_mapping.json")
locator = NoveltyLocator(mapper, cluster_gap_cm=30.0)

# Per-spectrum (inside predict()):
peaks            = extractor.extract_full(spectrum_preprocessed)   # list[Peak]
annotated_peaks  = mapper.annotate_peaks(peaks)                    # list[AnnotatedPeak]
disambig         = mapper.disambiguate_compound(annotated_peaks)   # dict
novelty          = locator.locate(peaks)                           # dict
```

**`predict()` return dict additions** (compose with T18 + T19 output):

```python
{
    # ... composition_mean, composition_std, ood_score, is_ood, etc from T18/T19 ...
    "peaks": [ap.to_dict() for ap in annotated_peaks],             # T20 + T21
    "likely_compounds_symbolic": disambig["likely_compounds"],
    "compound_votes": disambig["votes"],
    "unsupported_compounds": disambig["unsupported_compounds"],
    "unknown_peaks": novelty["unknown_peaks"],                     # T22
    "novelty_clusters": novelty["clusters"],
    "novelty_hints": novelty["hints"],
}
```

### D.3 Cross-check pattern for T26 report (now mandatory per P4AB-9)

The mixture-test finding makes this pattern mandatory, not optional. T26 reports MUST cross-check:

```python
# Pseudocode for T26
for cmp_name, (mean, std) in composition.items():
    in_learned  = mean > 0.05
    in_symbolic = cmp_name in disambig["likely_compounds"]
    if in_learned and not in_symbolic:
        flag = "learned says present, symbolic does not -- check if peaks were missed"
    elif in_symbolic and not in_learned:
        flag = "symbolic says present, learned does not -- possible false-positive vote"
    elif in_learned and in_symbolic:
        flag = "agreement: present"
    else:
        flag = "agreement: absent"
```

The row-780 mixture from local testing illustrates both error modes simultaneously:

| Compound | gt | Learned head (illustrative) | Symbolic likely? | Expected flag |
|---|---|---|---|---|
| Histidine | 0.45 | (high) | YES | agreement: present |
| Glucosamine | 0.19 | (medium-high) | NO | "learned says present, symbolic does not -- missed peaks" |
| Alanine | 0.15 | (medium) | NO | agreement: absent or "learned says present, symbolic does not" |
| Asparagine | 0.10 | (low) | NO | agreement: absent |
| Glutamic Acid | 0.08 | (low) | YES | "symbolic says present, learned does not -- false positive" |
| Aspartic Acid | 0.03 | (very low) | NO | agreement: absent |

This is the **interpretability story for the thesis defense**: the two heads disagree in characteristic ways, and the disagreement is itself useful information.

### D.4 Files on disk Phase B / Phase C can rely on

| File | Format | Source |
|---|---|---|
| `engine/bond_mapping.json` | JSON, 30 entries, **v1.1 tolerances** | Phase A-B (this chat) |
| `engine/peak_extractor.py` | importable | Phase A-B |
| `engine/symbolic_mapper.py` | importable | Phase A-B |
| `engine/novelty_locator.py` | importable | Phase A-B |
| `results/sanity/peak_demo_histidine.png` | PNG (synthetic) | Phase A-B sanity demo |
| `results/sanity/peak_demo_histidine_real.png` | PNG (real cache row) | Phase A-B local verification |
| `scripts/real_histidine_demo.py` | one-off verification script | Phase A-B local verification |
| `data/processed/wavenumbers.npy` | (1024,) float64 | Groundwork T03 |
| `data/processed/spectra_full.pt`, `labels.pt`, `vial_ids.npy` | preprocessed cache | Groundwork T04 |
| `data/splits/split_A_composition_ood.json`, `split_A_prime_sample_level.json` | indices | Phase A T07 |
| `checkpoints/best.pt` | torch dict (Phase 2 schema) | Chat 3 Phase 2 |
| `results/midcheckpoint_predictions.npz` | test-set predictions from T17 | Chat 2 Phase A |
| `results/ood_demo/calibration.json` | OODScorer state | Chat 3 stretch run |

### D.5 What Chat 2 Phase B (T23/T24/T25) needs to know

**Phase B does NOT consume any new Phase A-B module.** The engine layer (T20/T21/T22) is downstream of the model and runs at inference time only. T23 (PCA+SVM baseline) and T24 (1D-ResNet without physics) are pure quantification heads — they output composition; no peak extraction, no symbolic mapping, no novelty.

**But the test split MUST match.** Per `data/splits/split_A_composition_ood.json` (from Phase A T07), Phase B baselines should be evaluated on the same `test` index list that Chat 2 T17 used for the main model. Same metric definitions (`src/eval/metrics.py` from Phase A T09):

```python
from src.eval.metrics import (
    quantification_mae,
    identification_accuracy,
    reconstruction_cosine_similarity,
    constraint_violation_rate,
    ood_auroc,
)
```

For T25 comparison table, Phase B should compute these four metrics on each baseline:

| Metric | PCA+SVM | ResNet-only | Our model (from T17) |
|---|---|---|---|
| Quantification MAE | T23 fills | T24 fills | 0.0550 (per T17) |
| Identification accuracy (threshold 0.05) | T23 | T24 | T17 reported value |
| Reconstruction cosine median | N/A (no recon) | N/A (no recon) | 0.9698 (per T17) |
| Constraint violation rate | T23 (SVM probabilities should be simplex-projected to make this meaningful) | T24 (softmax output is already simplex) | 0.0000 (per T17) |

Note: reconstruction cosine and OOD AUROC are **only meaningful for our model** (the baselines have no reconstruction module and no OOD-tuned head). That's the point of the comparison — the baselines beat us on MAE *if they do*, but we win on the differentiator metrics. This is the headline of the thesis defense.

### D.6 What Chat 4 Phase C (T_glue/T26/T27) needs to do

Per the chat-task message (Phase C-D scope, original numbering):

| Task | File | Estimated | Notes |
|---|---|---|---|
| T_glue | `src/inference/predict.py` | 2-3h | Skeleton in D.2. Wraps Chat 3's T18/T19 + this chat's T20/T21/T22 + T26 report. |
| T26 | `src/inference/report.py` | 4h | Markdown template per chat-task message. **D.3 cross-check pattern is now mandatory** per P4AB-9. Show composition_std alongside means (Glucosamine high-uncertainty headline). |
| T27 | `src/inference/visualize.py` | 5h | Three functions: `plot_reconstruction_overlay`, `plot_peak_annotations` (colour-coded per compound; unmatched grey), `plot_ood_summary` (shaded novel regions + OOD gauge). |
| Demo | 3 reports in `results/reports/` | 1-2h | (1) ID test sample (composition close to ground truth), (2) Mild OOD (high-Glucosamine test sample -- expect wide variance), (3) Hard OOD (synthetic spike OR MoS2 if Phase D loader is ready). |
| T28 (opt) | `dashboard/app.py` | 6-8h | Streamlit 3-tab UI. Only if Phase C completes ahead of schedule. |

### D.7 Order of work for Chat 4 Phase C

1. **First**: drop these files into the project, run `pytest tests/ -v`. Must be 281/281 PASS.
2. **Second**: `peak_demo_histidine_real.png` already exists; visually verify Histidine peaks line up with v1.1 retuned DB entries.
3. **Third**: T_glue -> T27 -> T26 in that order (T26 needs visuals to embed; T27 needs T_glue output schema).
4. **Last**: 3 demo reports, **each illustrating the D.3 cross-check** (one with agreement, one with learned-says-yes / symbolic-no, one with OOD novelty cluster).

---

## E. Quick reproduction recipe (Chat 2 Phase B day 1 verification)

Phase B doesn't directly consume Phase A-B modules, but **must verify the engine tests still pass on the current cache** before starting baseline work. Otherwise a future regression in `bond_mapping.json` could silently break T_glue downstream.

```powershell
cd E:\Project\KhoaLuanCourse
.venv\Scripts\activate

# 1. Verify the full test suite is still green (one-time, ~40 sec)
pytest tests\ -v --tb=short
# Expected: 281 passed in ~40s

# 2. Verify the engine sees the cache correctly
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from engine.peak_extractor import PeakExtractor
from engine.symbolic_mapper import BondMapper

wn = np.load('data/processed/wavenumbers.npy')
X = torch.load('data/processed/spectra_full.pt', weights_only=True).numpy()
Y = torch.load('data/processed/labels.pt', weights_only=True).numpy()

# Find a high-Histidine row (will be useful for T25 qualitative analysis)
mask = Y[:, 4] > 0.95  # Histidine col is index 4
i = int(np.argmax(mask))
print(f'High-His row {i}: labels = {Y[i].round(3)}')

ext = PeakExtractor(wn)
mapper = BondMapper.from_json('engine/bond_mapping.json')
peaks = ext.extract_full(X[i])
ann = mapper.annotate_peaks(peaks)
d = mapper.disambiguate_compound(ann)
print(f'Likely compounds: {d[\"likely_compounds\"]}')
print(f'Votes: {{{\", \".join(f\"{k}: {v:.1f}\" for k,v in d[\"votes\"].items() if v > 0)}}}')
"
# Expected: 'Likely compounds: [..., \"Histidine\", ...]'  with His vote >= 3.0
# (the v1.1 retune ensures all 4 imidazole peaks match high-confidence)

# 3. Now safe to start T23 / T24
```

If step 1 or 2 fails, do NOT start Phase B baseline work — fix the engine regression first. The baselines themselves are independent of the engine, but the thesis's comparison story depends on the engine being trustable.

---

## F. Roadmap remaining

| Days | Chat | Tasks | Output |
|---|---|---|---|
| 11-12 | **Chat 2 Phase B (NEXT)** | T23 PCA+SVM, T24 ResNet-only, T25 comparison | `results/benchmark_table.md`, `results/benchmark_table.csv` |
| 12-13 | Chat 4 Phase C | T_glue + T26 + T27 + 3 demo reports | inference pipeline, thesis-defense demos |
| 13 | Chat 4 Phase D (stretch) | bacteria_ID + MoS2 loaders, T28 Streamlit dashboard | real OOD AUROC + dashboard |
| 14 | Chat 5 integration | REPORT.md, README quickstart, tag v0.1.0-mvp | thesis-ready repo |

---

## G. Read order for Chat 2 Phase B

1. **This file** (`CHAT4_PHASE_AB_HANDOVER.md` v1.1) — what Phase A-B produced + Phase B contracts (especially §D.5).
2. **`CHAT2_TASK17_HANDOVER.md`** — T17 verdict + per-compound MAE/r table. Headline numbers (MAE 0.0550, recon_cos 0.9698, CVR 0.0000) are the column for "Our model" in the T25 comparison.
3. **`CHAT3_CORE_ENGINE_HANDOVER.md` §D** — checkpoint loading recipe + model architecture (for T24 ResNet-only baseline: copy backbone + quant head, drop reconstruction + physics loss).
4. Custom Instructions §6 (targets) + §11 (Day-14 DoD) — to know what numbers count as "win" for T25.
5. Skim Custom Instructions §2 (out-of-scope list) — to avoid scope creep on baselines.

---

## H. Hand-off for Chat 4 Phase C (after Chat 2 Phase B finishes)

When Chat 2 Phase B is done, it should append a brief addendum to this handover (or write its own `CHAT2_PHASE_B_HANDOVER.md`) covering:

- `results/benchmark_table.{md,csv}` contents (3-row table for the 3 models)
- T23 / T24 checkpoint paths (`checkpoints/baselines/pca_svm.pkl`, `resnet_only_best.pt`)
- Any new metrics added beyond the 4 in §D.5
- Whether T25 found anything noteworthy in per-compound breakdown (e.g. does the ResNet-only baseline also struggle with Glucosamine, validating that the difficulty is in the data not our model?)

Then Chat 4 Phase C reads both this file and the Phase B addendum, and proceeds to T_glue / T26 / T27.

---

*Document version 1.1. Updated post-local-deployment by Chat 4 Phase A-B
(T20 + T21 + T22). Real-data verification on user's Windows machine
produced 281/281 tests passing and surfaced the v1.1 DB tolerance retune
(P4AB-8) plus the mixture-disambiguation finding (P4AB-9). Hand to Chat 2
Phase B for T23 + T24 + T25 next; then to Chat 4 Phase C for T_glue + T26
+ T27; also relevant to Chat 5 final integration.*
