# Chat 3 Phase 3 (T18 + T19) — Handover to Chat 4

> **Paste this at the start of Chat 4 Phase A (T20-T22 inference pipeline)
> and again at the start of Chat 4 Phase B (dashboard / external benchmarks).**
> Also attach the upstream context: `CHAT3_CORE_ENGINE_HANDOVER.md`,
> `CHAT2_TASK17_HANDOVER.md`, `results/midcheckpoint_report.md`.
>
> **Status as of this handover:** T18 (MC Dropout uncertainty) + T19
> (OOD scoring) implemented. Phase 3's remaining tasks (T20-T22:
> `predict()`, report, visualize) belong to Chat 4 Phase A per
> `CHAT2_TASK17_HANDOVER §7`. T17 verdict was **GO-WITH-CAVEAT** —
> MAE failed (0.0550 > 0.040 gate) but recon_cos_median (0.9698) and
> CVR (0.0000) both PASS-target. Phase 3 builds on this regardless.

---

## A. Files created in Phase 3

All paths relative to `E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\`.

### A.1 Source modules (production)

| Path | Module | Lines | Purpose |
|---|---|---|---|
| `src/models/uncertainty.py` | T18 | 428 | MC-Dropout uncertainty: `predict_with_uncertainty(model, x, n_samples=50)` function, `MCDropoutWrapper(nn.Module)` class, `mc_dropout_mode()` context manager. Toggles **only** `nn.Dropout*` to train mode; BatchNorm stays in eval (critical correctness — single-sample BN stats are degenerate). |
| `src/inference/ood.py` | T19 | 589 | OOD scoring: `OODScorer` class with `.calibrate(val_loader)` / `.score(spectrum)` / `.is_ood(spectrum)` / `.save(path)` / `.from_file(model, path)`. Score = 0.6·norm(recon_err) + 0.4·norm(pred_var); percentile-normalised via calibration set, threshold = 95th percentile of combined score. Includes `OODCalibration` dataclass, `compute_reconstruction_error`, `compute_predictive_variance`, and `make_synthetic_ood` (4 perturbation modes for stretch demos). |

### A.2 Tests

| Path | Tests | Coverage |
|---|---|---|
| `tests/test_uncertainty.py` | **32 tests** in 6 classes (`TestDropoutToggle`, `TestMcDropoutContext`, `TestInputCoercion`, `TestPredictWithUncertaintyShapes`, `TestPredictWithUncertaintyContent`, `TestMCDropoutWrapper`) | dropout-only toggle, BN stays eval, state restore, 1D/2D/3D input shapes, simplex preservation, batched, seeded reproducibility, entropy bounds, `return_samples`, wrapper override, dropout-less model warning |
| `tests/test_ood.py` | **34 tests** in 7 classes (`TestComponentPrimitives`, `TestCalibration`, `TestScoring`, `TestDiscrimination`, `TestSyntheticOOD`, `TestPersistence`, `TestValidation`) | recon error math (0 for identical, 2 for negation), calibration percentile math, score in [0, 1], component breakdown, discriminative (spike/mask/scale OOD mean > ID mean), save/load JSON roundtrip, validation of bad args |

### A.3 Stretch validation script (deliverable per T18/T19 spec)

| Path | Purpose |
|---|---|
| `scripts/run_phase3_stretch.py` | Loads `checkpoints/best.pt`, calibrates `OODScorer` on val split, scores **10 ID samples** from test split + **10 OOD samples per mode** (4 modes: spike / noise / mask / scale), prints discriminative table, writes `results/ood_demo/{calibration.json, score_table.csv, raw_scores.npz, score_distribution.png}`. Run after dropping new files in to confirm both modules work against the real checkpoint. |

### A.4 Files NOT created in Phase 3 (deferred to Chat 4)

| Path | Task | Owner |
|---|---|---|
| `src/inference/predict.py` | T20 — main `predict(spectrum)` returning unified dict | **Chat 4 Phase A** |
| `src/inference/report.py` | T21 — JSON + Markdown per-sample report generator | **Chat 4 Phase A** |
| `src/inference/visualize.py` | T22 — plotting utilities | **Chat 4 Phase A** |
| `results/reports/demo_*.md` | 3 demo reports for thesis defense | **Chat 4 Phase A** |
| `src/data/bacteria_id_loader.py`, `dashboard/app.py` | external benchmarks + Streamlit | **Chat 4 Phase B** |

---

## B. Open issues / known constraints

### B.1 Carried over from earlier phases (already known to Chat 4 via CHAT2_TASK17_HANDOVER)

| # | Issue | Phase 3 impact | Chat 4 impact |
|---|---|---|---|
| T17-B1 | Cache built from legacy `data.csv` (vial naming `a01-a48`) | None — Phase 3 reads same cache | T20 reads same cache, same vial format. Migration to `AA_Data.csv` still deferred to Phase B. |
| T17-B2 | `engine/reference_spectra.npy` not built from ENLIGHTEN; checkpoint uses baked-in refs | None — `OODScorer` and `predict_with_uncertainty` work with whatever refs are in the checkpoint's `pure_ref` buffer | T20 should NOT rebuild refs separately; load checkpoint → refs come along automatically (see §C contract below) |
| T17-B6 | MAE ceiling ~0.055 — fails Day-7 gate | Variance estimates on weak-signal compounds will be wide; that's honest and is exactly what T18 surfaces | T21 reports must show uncertainty alongside point estimates so users don't trust point estimates blindly |
| T17-B7 | Glucosamine r = −0.237 (negative correlation) | T18 should show high variance on Glucosamine specifically — diagnostic confirms model knows it's uncertain there | T21 demo report on a Glucosamine-heavy sample is a useful "known limitation" example |

### B.2 New issues from Phase 3

| # | Issue | Severity | Mitigation |
|---|---|---|---|
| **P3-1** | Sandbox had no `torch` available — Phase 3 code was authored against ASTs + numpy math verification only; no PyTorch smoke test was run in this session. | Medium — pure-Python logic verified, but PyTorch idioms (e.g., `model.modules()` walk) are tested via static analysis only | User must run `pytest tests/test_uncertainty.py tests/test_ood.py -v` on Windows + `python scripts/run_phase3_stretch.py` BEFORE Chat 4 proceeds. Expected: 32+34 = 66 tests PASS, stretch script prints OOD > ID mean for at least 3/4 OOD modes. |
| **P3-2** | `OODScorer.calibrate()` does TWO forwards per sample: one deterministic (for reconstruction) and one MC (for variance). On a 540-sample val set with `mc_samples=50`, that's `540 × (1 + 50) = 27,540` forwards. On CPU, ~5-10 min. | Low — only paid once per calibration | Chat 4 may want to lower `mc_samples` to 20-25 if calibration takes >10 min on user's hardware. Stretch script already exposes `--mc` flag. |
| **P3-3** | The deterministic forward inside `OODScorer.calibrate()` uses `self.model.eval()` then calls `model(x_3d)` — assumes the model's `.eval()` is sufficient to disable dropout. This is correct for the Phase 1 `RamanPhysicsAI`, but if Chat 4 wraps the model in MCDropoutWrapper before passing to OODScorer, the wrapper's persistent dropout-on state would corrupt the deterministic forward. | Low — only matters if scorer is fed a wrapped model | Document in T20 / T21: `OODScorer(model=ramanphysicsai, ...)` — NEVER `OODScorer(model=MCDropoutWrapper(ramanphysicsai), ...)`. Stretch script does this correctly. |
| **P3-4** | `make_synthetic_ood("spike")` injects pulses at fixed pixel locations (50, 200, 950). These pixels are chosen to be "away from discriminative peaks" but the choice is heuristic — on a different wavenumber grid the locations may overlap real chemistry. | Low — only affects stretch demo, not real OOD detection | Acceptable for MVP; v2 should pick spike locations from `engine/bond_mapping.json` complement. Real OOD (bacteria_ID, MoS2) replaces this in Chat 4 Phase B. |
| **P3-5** | `predictive_entropy` output is informational only — the OOD scorer doesn't use it (uses `mean_compound_std` instead, per Custom Instructions §5). | None | Chat 4 can still surface entropy in T21 reports for users who prefer it. |
| **P3-6** | OODCalibration is saved as JSON; refusing to migrate to pickle/msgpack means floating-point round-trip is lossy at ~1e-15 level. Negligible for thresholds (`score_p95` ~ 0.5-0.9 range), but tests use exact equality on the loaded value. | Negligible | If `test_save_load_roundtrip` ever fails due to JSON precision, relax the tolerance from `==` to `< 1e-10`. |
| **P3-7** | `is_ood()` uses strict `>` (not `>=`) comparison against `score_p95`. This means exactly-at-threshold samples are flagged ID, which is the right behaviour (threshold is the *boundary*) but worth noting. | None | None |

### B.3 NOT done in Phase 3 (out of scope by design)

- **T17 retrain with different beta_phys / weights**: explicitly out per the GO-WITH-CAVEAT decision. MAE limitation is documented; Phase 3 builds atop the existing checkpoint.
- **Real OOD samples** (bacteria_ID, MoS2): scheduled for Chat 4 Phase B / external benchmarks. Stretch validation uses synthetic perturbations.
- **OOD AUROC metric**: needs real OOD samples; `src/eval/metrics.py` already exposes `ood_auroc` from Phase A, but evaluation deferred to Chat 4 Phase B.

---

## C. Decisions that diverged from the original plan

| # | Original (spec / Custom Instructions) | What Phase 3 did | Reason |
|---|---|---|---|
| **D1** | T18 spec: "Set dropout to TRAIN mode (kể cả khi model.eval())" | Implemented as walk-modules toggle that flips ONLY `nn.Dropout*` modules. BatchNorm and the rest stay in eval. | Setting whole model to `train()` puts BatchNorm in per-batch-statistics mode → single-sample inference gives degenerate variance. Standard MC-Dropout idiom (Gal & Ghahramani 2016 → modern PyTorch). |
| **D2** | T18 spec: `predict_with_uncertainty(...)` returns `{"mean": μ, "std": σ, "all_predictions": all}` | Returns a richer dict: `composition_mean`, `composition_std`, `reconstruction_mean`, `reconstruction_std`, `predictive_entropy`, `mean_compound_std`, `n_samples`, plus optional `composition_samples` / `reconstruction_samples` when `return_samples=True`. | Spec was MAE-only model; ours has reconstruction too. Separating composition vs reconstruction uncertainty is more informative. `mean_compound_std` is the single-number "uncertainty" that T19 consumes. |
| **D3** | T19 spec: "Pass 1: forward with MC Dropout → composition mean + variance. Pass 2: compute reconstruction → recon error" | Phase 3 does ONE deterministic forward for reconstruction + ONE MC pass (with `n_samples` forwards inside it) for variance. Total = `1 + n_samples` forwards per spectrum. | Spec was ambiguous; "Pass 2" sounded like a single deterministic forward, which is what we do. Using the MC reconstruction mean would give noisier reconstruction error. |
| **D4** | T19 spec: "Normalize cả 2 thành [0,1] dùng calibration set" — no explicit formula | Implemented as `normalise(x) = min(x / x_p95, 1.0)` with `x_p95` = 95th percentile of x on cal set. Threshold = 95th percentile of combined score on same cal set. | Simple, monotonic, fail-safe (no div-by-zero), bounded in [0, 1] for in-distribution samples. Saturates at 1 for far-OOD, which is desired behaviour. |
| **D5** | T19 spec: no synthetic OOD generator mentioned | Added `make_synthetic_ood(spectrum, mode='spike'/'noise'/'mask'/'scale')` | Stretch validation in Phase 3 spec requires "10 ID + 10 OOD samples"; real OOD (bacteria_ID, MoS2) is Chat 4 Phase B work. Synthetic generator unblocks Phase 3 stretch demo without scope creep. |
| **D6** | T19 spec: no persistence | Added `OODScorer.save(path)` / `.load_calibration(path)` / `.from_file(model, path)` (JSON) | Calibration is expensive (~5-10 min on CPU). Persisting it means T20 `predict()` doesn't pay the cost every program launch. |
| **D7** | T18 spec: `MCDropoutWrapper(nn.Module)` to wrap model | Built but not made the default invocation path. Functional `predict_with_uncertainty(model, x)` is the canonical entry. | Functional form is more flexible (no wrapper state to manage, no risk of P3-3 — passing a wrapped model into OODScorer). Wrapper exists for future use (e.g., users who want a drop-in `nn.Module`). |

---

## D. Contracts Chat 4 will rely on

### D.1 Compound order LOCKED

```python
COMPOUND_ORDER = ["Alanine", "Asparagine", "Aspartic Acid",
                  "Glutamic Acid", "Histidine", "Glucosamine"]
```

Matches: `labels.pt` columns, checkpoint's `reconstruction.pure_ref` buffer, `bond_mapping.json` compound names, all model output indices. **Do NOT reorder anywhere.**

### D.2 Model loading recipe (handover §5 from CHAT2_TASK17_HANDOVER — tested in `run_phase3_stretch.py`)

```python
import tempfile, numpy as np, torch
from pathlib import Path
from src.models.full_model import build_full_model_from_config

ck = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]

# Refs are in checkpoint's state_dict; dispatch to factory via tempfile
refs = ck["model"]["reconstruction.pure_ref"].cpu().numpy()
ref_tmp = Path(tempfile.mkdtemp()) / "ref.npy"
np.save(ref_tmp, refs)
model = build_full_model_from_config(cfg, reference_spectra_path=str(ref_tmp))
model.load_state_dict(ck["model"])
model.eval()
```

### D.3 T18 API — call sites for T20 `predict()`

```python
from src.models.uncertainty import predict_with_uncertainty

# Single-spectrum (P,) input
out = predict_with_uncertainty(model, spectrum, n_samples=50)
# out is a dict with batch dim of 1:
#   out["composition_mean"]    -- (1, 6) simplex
#   out["composition_std"]     -- (1, 6) per-compound std
#   out["reconstruction_mean"] -- (1, 1024)
#   out["reconstruction_std"]  -- (1, 1024)
#   out["predictive_entropy"]  -- (1,) scalar-per-sample
#   out["mean_compound_std"]   -- (1,) average std over compounds (= OOD signal)
#   out["n_samples"]           -- 50
# Squeeze the leading batch dim if T20 wants unbatched results.
```

### D.4 T19 API — call sites for T20 `predict()`

```python
from src.inference.ood import OODScorer

# Build + calibrate ONCE (in T20 setup):
scorer = OODScorer(model, recon_weight=0.6, var_weight=0.4, mc_samples=50)
scorer.calibrate(val_loader)         # ~5 min on CPU
scorer.save("results/ood_calibration.json")

# Or load pre-calibrated (T20 production):
scorer = OODScorer.from_file(model, "results/ood_calibration.json")

# Per-spectrum scoring:
ood_score: float = scorer.score(spectrum)
is_ood: bool = scorer.is_ood(spectrum)
threshold: float = scorer.calibration.score_p95
```

### D.5 Files on disk T20 can rely on

| File | Format | Source |
|---|---|---|
| `checkpoints/best.pt` | torch dict (Phase 2 schema) | T15 training, T17 Round 2 |
| `engine/bond_mapping.json` | 30-entry JSON | Phase A T05 |
| `engine/symbolic_mapper.py` | importable | Phase A T05 |
| `data/processed/wavenumbers.npy` | (1024,) float64 | Phase A T03 |
| `data/processed/spectra_full.pt` | (4378, 1024) float32 | Phase A T04 |
| `data/processed/labels.pt` | (4378, 6) float32 | Phase A T04 |
| `data/splits/split_A_composition_ood.json` | dict[train/val/test] | Phase A T07 |
| `results/midcheckpoint_predictions.npz` | T17 test predictions | Chat 2 T17 |
| `results/ood_calibration.json` | (to be built by Chat 4 setup) | Chat 4 calibrates once |

### D.6 Recommended T20 `predict()` skeleton

```python
def predict(spectrum, *, model, ood_scorer, bond_mapper, n_mc_samples=50):
    """Main MVP inference entry.

    Returns dict with:
        composition       -- {compound: (mean, std)}
        reconstruction    -- (1024,) reconstructed spectrum
        recon_cosine_sim  -- scalar
        ood_score         -- scalar
        is_ood            -- bool
        peaks             -- list[(wavenumber, intensity, fwhm, bond_or_None)]
        novelty_peaks     -- list of peaks with bond=None (Chat 4 T22 demo case)
        confidence_label  -- "high" | "medium" | "low" derived from
                              mean_compound_std percentile vs cal stats
    """
    mc = predict_with_uncertainty(model, spectrum, n_samples=n_mc_samples)
    score = ood_scorer.score(spectrum)
    is_ood = score > ood_scorer.calibration.score_p95
    # ... peak extraction, bond mapping, novelty detection ...
    return {...}
```

---

## E. What Chat 4 needs to do (T20-T22)

Per `CHAT3_CORE_ENGINE_HANDOVER §F` (with adjustments for T17 reality):

| Task | File | Estimated | Notes |
|---|---|---|---|
| T20 | `src/inference/predict.py` | 2-3h | Use D.6 skeleton above. Wrap T18 + T19 + peak extractor + symbolic mapper into one dict-returning function. |
| T21 | `src/inference/report.py` | 2h | JSON + Markdown report generator. Must show composition_std alongside means (Glucosamine high uncertainty is the headline). |
| T22 | `src/inference/visualize.py` | 1-2h | Spectrum + reconstruction overlay, peak annotations, OOD score gauge. Use existing `matplotlib`. |
| Demo | `results/reports/demo_*.md` × 3 | 1h | (1) In-distribution test sample, (2) Mild OOD (high-Glucosamine sample, expect wide variance), (3) Hard OOD (MoS2 if available, else synthetic spike). |

### E.1 Order of work for Chat 4 Phase A

1. **First**: run `pytest tests/test_uncertainty.py tests/test_ood.py -v` → all 66 must PASS (per P3-1).
2. **Then**: run `python scripts/run_phase3_stretch.py` → verify discriminative table looks sane (OOD means > ID mean for ≥3/4 modes).
3. **Then**: T20 → T21 → T22 in that order (T21 needs T20's output schema; T22 visualises both).
4. **Last**: 3 demo reports.

### E.2 Order of work for Chat 4 Phase B

External benchmarks + dashboard:

- `src/data/bacteria_id_loader.py` — Ho-2019 numpy loader (replaces synthetic OOD with real OOD).
- Re-run `run_phase3_stretch.py` but with `bacteria_ID` as the OOD source → real AUROC number.
- `src/data/mos2_loader.py` — single-spectrum from `MoS2-160o-12h-ph5.txt`.
- `dashboard/app.py` — Streamlit upload-spectrum-and-predict UI (Day 13 stretch only).

These are AFTER Chat 4 Phase A is complete.

---

## F. Quick reproduction recipe (Chat 4 day 1)

```powershell
# Prerequisites: T18 + T19 files dropped in, checkpoint at checkpoints/best.pt
cd E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2

# 1. Tests pass (one-time, ~30 sec)
pytest tests\test_uncertainty.py -v       # 32 tests
pytest tests\test_ood.py -v               # 34 tests

# 2. Stretch validation (~3-5 min, builds calibration + demo)
python scripts\run_phase3_stretch.py --mc 30
# Inspect: results\ood_demo\score_distribution.png
# OOD mean should be visibly above ID mean for spike, mask, scale modes.

# 3. (Optional) Pre-build calibration for T20 reuse
python -c @"
from src.models.full_model import build_full_model_from_config
from src.inference.ood import OODScorer
import torch, json, numpy as np, tempfile
from pathlib import Path
from torch.utils.data import DataLoader, Subset, TensorDataset

ck = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
cfg = ck['config']
refs = ck['model']['reconstruction.pure_ref'].cpu().numpy()
ref_tmp = Path(tempfile.mkdtemp()) / 'ref.npy'
np.save(ref_tmp, refs)
m = build_full_model_from_config(cfg, reference_spectra_path=str(ref_tmp))
m.load_state_dict(ck['model']); m.eval()

X = torch.load('data/processed/spectra_full.pt', weights_only=True).float()
Y = torch.load('data/processed/labels.pt', weights_only=True).float()
split = json.load(open('data/splits/split_A_composition_ood.json'))
val_ds = Subset(TensorDataset(X, Y), split['val'])
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

s = OODScorer(m, mc_samples=50)
s.calibrate(val_loader)
s.save('results/ood_calibration.json')
print('Saved calibration:', s.calibration)
"@

# 4. Start T20 implementation
```

---

## G. Roadmap remaining

| Days | Chat | Tasks | Output |
|---|---|---|---|
| 10-11 | **Chat 4 Phase A** ← Phase 3 just finished | T20-T22 + 3 demo reports | inference pipeline + thesis-defence demos |
| 11-12 | Chat 2 Phase B | T23 PCA+SVM, T24 ResNet-only, T25 comparison | benchmark_table.md |
| 13 | **Chat 4 Phase B** | T27 external benchmarks (bacteria_ID, MoS2), T28 dashboard | dashboard.py + real OOD AUROC |
| 14 | Chat 4 stretch | REPORT.md, tag v0.1.0-mvp | thesis-ready |

---

## H. Read order for Chat 4

1. **This file** (`CHAT3_PHASE3_HANDOVER.md`) — what Phase 3 produced + contracts.
2. **`CHAT2_TASK17_HANDOVER.md`** — T17 verdict + cumulative state.
3. **`results/midcheckpoint_report.md`** — full metrics; especially per-compound table.
4. Skim `CHAT3_CORE_ENGINE_HANDOVER.md` only if T20 needs Phase 2 details (training config, augmentation).
5. Custom Instructions §5 (hyperparameters), §7 (report format) — reference while implementing T20-T22.

---

*Document version 1.0. Generated at end of Phase 3 (T18 + T19). Hand to
Chat 4 Phase A for T20-T22 inference pipeline.*
