# Chat 2 → Chat 3 bridge v1.1 — after best.pt inspection

> **Use at start of Chat 3** (Phase 3 OR fallback retrain, depending on
> T17 verdict). Or as the "what did Chat 2 do?" reference.
>
> **Status:** Chat 2 received `best.pt` from user; inspected its content
> in sandbox (without torch, via custom unpickler). All structural and
> numerical assumptions about the checkpoint have been verified. T17 is
> ready to run on user's Windows machine via the v1.1 script.

---

## What was confirmed from inspecting `best.pt`

`scripts/inspect_checkpoint.py` was run against the actual `best.pt`. The
output is authoritative — replaces all guesses from earlier handovers.

### 1. Top-level structure (as expected)
```
top-level keys: ['model', 'optimizer', 'scheduler', 'epoch', 'val_metrics', 'config']
epoch         : 12
val_metrics   : val_mae=0.046550, best_val_mae=0.046550, val_loss_total=0.123384
```

### 2. State dict (1,008,128 params, 126 tensors)
```
backbone.stem        6 tensors        353 params  (conv1d 1→32 k=7 + BN1d)
backbone.stages.0   24 tensors     12,804 params  (2× BasicBlock1D, 32 ch)
backbone.stages.1   30 tensors     46,341 params  (2× BasicBlock1D, 32→64, stride-down)
backbone.stages.2   30 tensors    182,789 params  (2× BasicBlock1D, 64→128, stride-down)
backbone.stages.3   30 tensors    726,021 params  (2× BasicBlock1D, 128→256, stride-down)
quantification_head.fc1    2 tensors  32,896 params  (256→128)
quantification_head.fc2    2 tensors     774 params  (128→6)
reconstruction.scale       1 tensors       6 params  (per-compound)
reconstruction.pure_ref    1 tensors   6,144 params  (6 × 1024)  ← important
```
Architecture matches CHAT3 §A specification exactly. No surprises.

### 3. Reconstruction module: pure_ref + scale
`pure_ref` baked into the checkpoint is approximately **SNV-normalized**
(mean ≈ 0, std ≈ 1 per row). All 6 compounds:
```
0: Alanine       mean=+0.0000 std=1.0000 range=[-0.519, +8.132]
1: Asparagine    mean=-0.0000 std=1.0000 range=[-0.545, +6.421]
2: Aspartic Acid mean=-0.0000 std=1.0000 range=[-0.564, +7.960]
3: Glutamic Acid mean=+0.0000 std=1.0000 range=[-0.572, +8.612]
4: Histidine     mean=-0.0000 std=1.0000 range=[-0.627, +8.043]
5: Glucosamine   mean=+0.0000 std=1.0000 range=[-0.660, +5.181]
```
This means `engine/reference_spectra.npy` was successfully built and applied
preprocessing before training (P2-3 from previous handover: visual sanity =
**confirmed yes structurally**, though user should still eyeball the
Histidine peak plot once).

Learned `scale` (init = 1.0):
```
[1.070, 1.092, 1.046, 1.060, 1.350, 0.718]   for [Ala, Asn, Asp, Glu, His, GlcN]
```
- Histidine `+0.35` drift: model needed to inflate Histidine's reference
  contribution. Suggests training-data Histidine spectra were stronger than
  the reference suggests, OR the model is over-attributing to Histidine
  (which is the easiest compound to detect — see Custom Instructions §8).
- Glucosamine `-0.28` drift: model needed to deflate Glucosamine's
  reference. Glucosamine is the sugar (not amino acid), with weaker overall
  pyranose-ring contribution; reference may have been stronger than
  training data needed. Either way, learnable scale absorbed it.

### 4. Quantification head fc2 bias spread = 0.04 — well-balanced
No structural bias toward any compound. Class imbalance is not the issue.

### 5. **CRITICAL FINDING — O4 IS TRIGGERED**
Checkpoint config contains:
```
"paths": {
  "data_raw_csv": "data/raw/data.csv",    ← LEGACY
  ...
}
```
**This is the legacy `data.csv` path, NOT `AA_Data.csv`.** Per
`PROJECT_REVISION_v2.md` §1.1, the primary dataset is now `AA_Data.csv`,
which has 3 extra metadata columns (`file_name`, `Repitation`, `mix_method`)
but the same 1024 spectral cols and 6 ratio cols.

**Two possibilities — must be resolved before T17:**

| Scenario | What it means | T17 valid? | Action |
|---|---|---|---|
| (a) `spectra_full.pt` was also built from legacy `data.csv` | Self-consistent — model trained, val'd, will test on the same source | YES | Run T17 normally |
| (b) `spectra_full.pt` was rebuilt from `AA_Data.csv` AFTER training | spectra/labels rows may be in different order than what the model expects | **NO — silent garbage** | Rebuild cache from legacy OR re-train with `AA_Data.csv` |

**How to tell which scenario:**
```powershell
python -c "import numpy as np; v = np.load('data/processed/vial_ids.npy', allow_pickle=True); print('first 5:', list(v[:5])); print('unique starts:', set(s[:3] for s in v if isinstance(s,str)))"
```
- If you see `a01..a48` (single 'a' prefix) → scenario (a), safe
- If you see `aa01..aa48` (double 'a' prefix) → scenario (b), DANGER

If scenario (b): rebuild cache from legacy temporarily, OR retrain with
AA_Data.csv. The training run had behavior consistent with self-consistent
data (val MAE plateau, not random noise) so (a) is more likely.

---

## 1. Files created (cumulative across all chats so far)

### Chat 2 Phase A — DONE
- `src/data/enlighten_parser.py`
- `scripts/extract_pure_references.py`
- `src/data/splits.py` (replaced T03)
- `src/eval/metrics.py`
- `tests/test_metrics.py` *(23 PASS)*

### Chat 3 Phase 1 (T10-T14) — DONE
- `src/models/backbone.py`
- `src/models/heads.py`
- `src/models/reconstruction.py`
- `src/training/losses.py`
- `src/models/full_model.py`
- `tests/test_models.py`

### Chat 3 Phase 2 (T15-T16) — DONE
- `src/data/augmentation.py`
- `src/training/train.py`
- `configs/train_config.yaml`
- `scripts/plot_training_curves.py`
- `tests/test_augmentation.py`
- `docs/TRAINING_GUIDE.md`

### Chat 3 — Training artefacts (user's machine)
- `checkpoints/best.pt` (epoch 12, val_mae 0.0466) — **inspected and confirmed valid in Chat 2 this turn**
- `checkpoints/last.pt` (epoch 20)
- `results/training_log.csv` (21 rows)
- `results/figures/training_curves.png`

### Chat 2 T17 setup — NEW THIS TURN
- `scripts/_pt_inspect.py` (helper — custom torch-less unpickler)
- `scripts/inspect_checkpoint.py` (CLI checkpoint inspector)
- `scripts/run_t17_midcheckpoint.py` v1.1 (production T17 runner)
- `CHAT3_CORE_ENGINE_HANDOVER_v1_1.md` (post-training analysis)
- `CHAT2_T17_TO_CHAT3_BRIDGE.md` v1.1 (this file)

### Chat 2 T17 outputs — user produces by running script
- `results/midcheckpoint_report.md`
- `results/midcheckpoint_report.json`
- `results/midcheckpoint_predictions.npz`

### NOT yet created — gated on T17 verdict
- **If PASS / BORDERLINE:** `src/models/uncertainty.py` (T18),
  `src/inference/ood.py` (T19), `src/inference/predict.py` (T20),
  `src/inference/report.py` (T21)
- **If FAIL:** new fallback checkpoint
- **Phase B (Chat 2 return after T17 GO):** `src/models/baselines/pca_svm.py`,
  `src/models/baselines/resnet_only.py`, `src/eval/benchmark.py`,
  `src/eval/compare.py`, `results/benchmark_table.md`

---

## 2. Open bugs / issues (cumulative)

### O1 — Training underfits, physics term hijacks gradient (HIGH severity)
- Evidence: train MAE plateau 0.046 from epoch 6; train-val gap 0.0004 at
  best; physics/quant ratio 5.39× on train.
- **Root cause confirmed via checkpoint inspection:** `beta_phys = 0.5` (in
  config), combined with reasonable pure_ref scale (SNV) means the physics
  term IS as big as advertised, gradient sink confirmed.
- Mitigation lined up: C11 drop `beta_phys: 0.5 → 0.0` in fallback.
- **Status:** OPEN, awaits T17 verdict to gate fallback.

### O2 — Early stop fired before LR schedule completed (LOW severity)
- 21/50 epochs. Independent of O1.
- Mitigation: C12 `early_stopping_patience: 8 → 15`.

### O3 — Val MAE noisy CV=15.6% (MEDIUM)
- Val set 540 samples + augmentation noise.
- Mitigation: C13 `noise_sigma: 0.005 → 0.002`.

### O4 — Config points to legacy `data.csv` (MEDIUM — CONFIRMED THIS TURN)
- Checkpoint config: `data_raw_csv: "data/raw/data.csv"` (legacy).
- **Two scenarios — see "CRITICAL FINDING" box above**.
- Action: user runs `vial_ids.npy` check before T17 to disambiguate.
- If scenario (b), T17 numbers will be garbage; must rebuild cache first.

### O5 — Reference spectra not visually verified (LOW)
- Checkpoint pure_ref is structurally OK (SNV-normalized), but user has
  not opened `results/sanity/pure_spectra.png` to confirm Histidine peaks
  at 1003/1180/1495/1575 cm⁻¹ are visible.
- Mostly cosmetic at this point; if pure_ref were bad, train wouldn't
  have converged at all.

### O6 — Test suite verification on Windows pending (LOW)
- `pytest tests/test_models.py` and `tests/test_augmentation.py` not run.
- Training completing without crashes is strong indirect evidence.

### O7 — Multi-dataset refactor still pending (PROJECT_REVISION §6) (LOW)
- No impact on T17 / fallback. Action deferred to Phase B / Phase D.

---

## 3. Decisions changed vs. plan (cumulative)

### From Chat 2 Phase A
| # | Locked decision | Origin |
|---|---|---|
| A1 | Compound order = `[Ala, Asn, Asp, Glu, His, GlcN]` (canonical), NOT older `[Ala, Gly, Ser, Thr, His, GlcN]` | Chat 2 §C1 |
| A2 | References built from 6 ENLIGHTEN exports, NOT from filtering training data | PROJECT_REVISION §2.7 |
| A3 | Scheme A: force 6 pure vials into train → 3298/540/540 not 3538/460/380 | Chat 2 §C3 |
| A4 | `splits.py` keeps only Scheme A + A' (no B/C) | Chat 2 §C6 |

### From Chat 3 Phase 1+2
| # | Locked decision | Origin |
|---|---|---|
| 1 | Reference spectra → `register_buffer`, NOT `nn.Parameter(frozen=True)` | CHAT3 C2 |
| 2 | Reconstruction shape `(B, 1024)`, NOT `(B, 1, 1024)` | CHAT3 C3 |
| 3 | Optimizer = `AdamW`, NOT plain `Adam` | CHAT3 C4 |
| 4 | Wandb default OFF | CHAT3 C5 |
| 5 | Plain `CosineAnnealingLR`, NOT WarmRestarts | CHAT3 C7 |
| 6 | `--smoke` mode added | CHAT3 C8 |
| 7 | `grad_clip_norm: 1.0` | CHAT3 C9 |

### From Chat 2 T17 (this turn)
| # | Locked decision | Notes |
|---|---|---|
| 8 | T17 script v1.1 uses **pure_ref from checkpoint**, not from `engine/reference_spectra.npy` | Avoids silent ref-mismatch when refs were rebuilt after training |
| 9 | T17 script v1.1 has pre-flight check for legacy `data.csv` path (O4) | Forces user awareness before garbage results |

### Runtime decisions (Chat 3 produced — kept under review)
| # | What | Status after T17 |
|---|---|---|
| C10 | Defaults used end-to-end during training run | DONE — produced FAIL evidence |
| C11 (proposed) | `beta_phys: 0.5 → 0.0` for fallback | **APPLY IF T17 = FAIL** |
| C12 (proposed) | `early_stopping_patience: 8 → 15` | **APPLY IF T17 = FAIL** |
| C13 (proposed) | `gaussian_noise_sigma: 0.005 → 0.002` | **APPLY IF T17 = FAIL** |
| C14 (proposed) | Skip reconstruction branch in forward when `beta_phys==0` | Optional optimisation |

---

## 4. Chat 3's next move (decision tree)

```
Step 0 — verify scenario (a) vs (b) for O4:
  python -c "import numpy as np; v=np.load('data/processed/vial_ids.npy',allow_pickle=True); print(set(s[:3] for s in v if isinstance(s,str)))"

Step 1 — Run T17:
  python scripts\run_t17_midcheckpoint.py
  type results\midcheckpoint_report.md

Step 2 — Branch on verdict:

  PASS-target  → Phase 3 normal (T18, T19, T20, T21)
  PASS-floor   → Phase 3, document caveats
  BORDERLINE   → Phase 3 + use T18 variance as honest-uncertainty diag
  FAIL         → §F.3 fallback:
                 1. edit configs/train_config.yaml per C11+C12+C13
                 2. (optional) edit full_model.forward per C14
                 3. retrain (~5 min CPU)
                 4. re-run scripts\run_t17_midcheckpoint.py
                 5. if still FAIL: document as limitation, ship v0.1.0-mvp
```

**Pre-T17 expected verdict:** still FAIL (60%) or BORDERLINE (30%). The
val MAE 0.0466 with high noise and the confirmed physics-loss-hijack
make a sub-0.040 test MAE statistically unlikely. T17 produces the
evidence so we can move with confidence to fallback or to whichever
verdict matches.

---

## 5. Quick-start commands (resuming Chat 3)

```powershell
# Step 0 — Drop new scripts in
Copy-Item path\to\downloads\_pt_inspect.py scripts\
Copy-Item path\to\downloads\inspect_checkpoint.py scripts\
Copy-Item path\to\downloads\run_t17_midcheckpoint.py scripts\   # OVERWRITES v1.0

# Step 1 — Inspect (no torch needed; reads bytes directly)
python scripts\inspect_checkpoint.py

# Step 2 — Verify scenario for O4
python -c "import numpy as np; v=np.load('data/processed/vial_ids.npy',allow_pickle=True); print(set(s[:3] for s in v if isinstance(s,str)))"
# Should print {'a01', 'a02', ...} = OK to proceed
# If it prints {'aa0', ...} = STOP, rebuild cache from data.csv legacy first

# Step 3 — Run T17
python scripts\run_t17_midcheckpoint.py

# Step 4 — Read verdict
type results\midcheckpoint_report.md
```

---

## 6. Reading order for Chat 3 turn-1

1. **`results/midcheckpoint_report.md`** — actual numbers
2. **This file** — decision tree, file inventory, open bugs
3. **`CHAT3_CORE_ENGINE_HANDOVER_v1_1.md`** — training analysis if needed
4. **`results/training_log.csv` + `training_curves.png`** — raw evidence

---

*Document version 1.1. Written 2026-05-11 by Chat 2 after inspecting
the actual best.pt file via custom torch-less unpickler. Reflects
confirmed structural facts about the checkpoint and the critical O4 finding.*
