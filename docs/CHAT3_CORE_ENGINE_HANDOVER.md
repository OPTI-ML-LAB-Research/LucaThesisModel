# Chat 3 (Core Engine) — Handover Summary

> **Use this document at the start of Chat 2 (return trip) to run T17
> mid-checkpoint, and any subsequent return to Chat 2 for T23/T24
> baselines and T25 comparison.**
>
> **Status:** Phase 1 (T10-T14) + Phase 2 (T15-T16) code DONE. Training
> run has been executed locally; results are recorded below in the
> "Actual training run" section. T17 (mid-checkpoint, Chat 2) is the
> next task.
>
> **Edit before pasting to Chat 2.** Sections marked
> `[FILL AFTER RUN]` need the real numbers from your training output;
> sections marked `[REVIEW]` need a yes/no confirmation. Everything else
> is locked-in code state.

---

## A. Files created in Phase 1 + Phase 2

All paths are relative to `E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\`.

### Phase 1 — Architecture (T10-T14)

| Path | Module | Purpose |
|---|---|---|
| `src/models/backbone.py` | T10 | `ResNet1DBackbone` + `BasicBlock1D`. 4 stages [32, 64, 128, 256], 2 blocks/stage. Input (B, 1, 1024) → output (B, 256). ~963K params. |
| `src/models/heads.py` | T11 | `QuantificationHead`. Linear(256→128)→ReLU→Dropout(0.2)→Linear(128→6)→Softmax. Output is a valid simplex. ~33K params. |
| `src/models/reconstruction.py` | T12 | `ReconstructionModule`. References as `register_buffer` (not Parameter), per-compound `scale` as `nn.Parameter` (init 1.0, learnable). Output (B, 1024). |
| `src/training/losses.py` | T13 | `quantification_loss` (MAE), `physics_loss` (MSE + λ·cosine_dist), `l2_regularization`, `combined_loss`. Returns dict with components for logging. |
| `src/models/full_model.py` | T14 | `RamanPhysicsAI`. Composes the three. Forward returns `{"composition", "reconstruction", "feature"}`. Includes `build_full_model_from_config` factory. |
| `tests/test_models.py` | new | 32 pytest cases across 5 classes (`TestBackbone`, `TestQuantificationHead`, `TestReconstructionModule`, `TestLosses`, `TestFullModel`). |

### Phase 2 — Training (T15-T16)

| Path | Module | Purpose |
|---|---|---|
| `src/data/augmentation.py` | T16 | `RamanAugmentation` (shift/scale/noise) + `AugmentedDataset` wrapper. Per-sample independent draws, seed-controlled. |
| `src/training/train.py` | T15 | End-to-end training script. Argparse, config merge, dataloaders, AdamW, cosine LR, train/val loop, early stopping, CSV logging, checkpointing, optional wandb, `--smoke` mode. |
| `configs/train_config.yaml` | new | Training-specific hyperparameters. Merged on top of `default.yaml`. |
| `scripts/plot_training_curves.py` | helper | Standalone plotter for `training_log.csv`; works on partial CSVs. |
| `tests/test_augmentation.py` | new | 20 pytest cases. |
| `docs/TRAINING_GUIDE.md` | doc | The runbook used to produce the training results in Section D. |

### Files NOT created in Chat 3 (deferred / out of scope)

- `src/models/uncertainty.py` (T18) — Phase 3, AFTER Day-7 gate
- `src/inference/ood.py` (T19) — Phase 3, AFTER Day-7 gate
- `src/inference/predict.py` (T20+) — Phase 3 / inference task
- `src/inference/report.py` (T21+) — Phase 3 / inference task
- `src/models/baselines/*` (T23, T24) — Chat 2 work after T17

---

## B. Open issues / known constraints

### B.1 — From Phase 1 (carried over)

**P1-1 (resolved if smoke + tests pass on user machine).** Sandbox where
this code was authored had no torch; verification was static + numpy-only.
User confirmed by running:
- `pytest tests/test_models.py` → **[FILL AFTER RUN]** out of 32 PASSED
- `pytest tests/test_augmentation.py` → **[FILL AFTER RUN]** out of 20 PASSED

**P1-4 (open, monitored during training).** Reconstruction `scale`
init=1.0 assumes `pure_ref` and preprocessed AA spectra are on the same
scale (both SNV-normalised). If they aren't, training compensates via
the `scale` parameter but converges slower. Watch `train_loss_physics`
trajectory: should drop steadily after epoch 3-5. **[FILL AFTER RUN]**:
ratio of physics/quant loss at epoch 10 = ___.

### B.2 — From Phase 2

**P2-2 (resolved by user before training).** Cache `data/processed/spectra_full.pt`
must be from `AA_Data.csv`, not legacy `data.csv`. Verified via vial naming
(`aa01`-`aa48`, not `a01`-`a48`).
**[REVIEW]** confirmed cache rebuilt from `AA_Data.csv` before training:
**[Yes / No]**

**P2-3 (resolved before training).** `engine/reference_spectra.npy` was
built via `extract_pure_references.py`. Histidine peaks visually
verified at ~1003/1180/1495/1575 cm⁻¹ in `results/sanity/pure_spectra.png`.
**[REVIEW]** visually confirmed: **[Yes / No / Partial — describe]**.

**P2-7 (open, monitor in training log).** Physics loss may dominate
combined loss in early epochs, pushing gradient toward `scale` only and
delaying quantification learning. Diagnostic in `TRAINING_GUIDE.md` §4.7.
**[FILL AFTER RUN]** — physics dominated? **[Yes / No]**. If yes,
mitigation applied: ___.

### B.3 — New issues discovered during this training run

**[FILL AFTER RUN]** — list anything that surfaced. Examples:
- "Gradient clipping fired aggressively at epoch 1; raised `grad_clip_norm` to 5.0"
- "Loss NaN at epoch 3; lowered `beta_phys` to 0.05 and resumed"
- "Training crashed at epoch 12 due to disk full; resumed with `--resume`"
- (none) — leave the list empty if smooth

---

## C. Decisions changed vs. original plan

| # | Original plan | What Chat 3 actually did | Reason |
|---|---|---|---|
| C1 | (Phase 1 D1) Compound order: Glycine, Serine, Threonine in T10/T11 spec | Canonical order: Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine | T10 spec was stale; canonical order matches `AA_Data.csv`, `engine/reference_spectra.npy`, and CHAT2_PHASE_A_HANDOVER decision C1 |
| C2 | (Phase 1 D2) Reference spectra "as `nn.Parameter(frozen=True)`" | `register_buffer` instead | PyTorch idiom: non-trainable tensors that move with `.to(device)` belong in buffers, not in `parameters()` |
| C3 | (Phase 1 D3) Reconstruction output `(B, 1, 1024)` | `(B, 1024)` (no channel dim) | Cleaner; physics_loss handles both 2D and 3D inputs uniformly |
| C4 | (Phase 2 D1) Optimizer = `Adam` (Custom Instructions §5) | `AdamW` | model05.py uses AdamW; properly decouples weight decay. lr/wd values kept per Custom Instructions |
| C5 | (Phase 2 D2) Wandb integration spec'd | Default OFF, opt-in via `use_wandb: true` + tool installed | CHAT2_PHASE_A_HANDOVER decision: skip wandb to repay 2h budget for dataloader refactor |
| C6 | (Phase 2 D5) Plot loss curves inline in train.py | Tearout to `scripts/plot_training_curves.py` | Decoupled = re-runnable after a crash; doesn't need torch to plot |
| C7 | (Phase 2 D6) Cosine annealing | Plain `CosineAnnealingLR`, NOT WarmRestarts (model05 used T_0=10 + T_mult=2) | Plain cosine is smoother for the 50-epoch budget; warm-restarts shine on longer runs |
| C8 | (Phase 2 D7) No "smoke" mode in spec | Added `--smoke` flag (2 epochs on 200/50 subset) | Critical-path safety: catches pipeline bugs in <2 min before committing 30-90 min full run |
| C9 | (Phase 2 D8) No grad clipping in spec | Default `grad_clip_norm: 1.0` | Cheap safety against physics-loss spikes when reconstruction is poor in early epochs |

**[FILL AFTER RUN]** — **runtime decisions** added during the actual
training run (e.g. "lowered beta_phys to 0.1 after observing physics
loss explosion at epoch 2"). Leave empty if defaults worked.

| C-extra | What was changed at runtime | Reason | Result |
|---|---|---|---|
| (e.g. C10) | (e.g. `loss.beta_phys: 0.5 → 0.1`) | (e.g. "epoch 2 NaN") | (e.g. "stable thereafter") |

---

## D. Actual training run

### D.1 Environment

- **Date / time:** 2026-05-11T09:37:55
- **Machine:** Windows-11-10.0.26200-SP0
- **Python:** 3.13.7
- **PyTorch:** 2.11.0+cpu
- **Device used:** cpu

### D.2 Commands actually run (in order)

```powershell
# Smoke
python -m src.training.train --config configs/train_config.yaml --smoke

# Full
python -m src.training.train --config configs/train_config.yaml

# (Any --resume calls or runtime config edits go here)
```

### D.3 Headline numbers

| Metric | Value | Status vs target |
|--|---|---|
| Best val MAE | **0.0466** | target ≤ 0.020 / floor ≤ 0.025 → **FAIL** |
| Best epoch | 12 | (which epoch saved `best.pt`) |
| Final epoch trained | 20 | of 21 total epochs in log |
| Total wall time | 3.9 min | (sum of `epoch_seconds`) |
| Train MAE at best epoch | 0.0462 | |
| Train/val MAE gap at best | 0.0004 | overfitting indicator (positive = val worse) |
| Val physics loss at best | 0.1537 | reconstruction quality |
| physics/quant ratio @ ep10 | 5.39 | P2-7 diagnostic; > 5 means physics dominates |

### D.4 Training log CSV — first / last 5 rows

Paste from `results/training_log.csv`:

```csv
epoch,lr,train_loss_total,train_loss_quant,train_loss_physics,train_mae,val_loss_total,val_loss_quant,val_loss_physics,val_mae,best_val_mae,epoch_seconds
0,0.0009990143508499217,46.528556548443035,0.08192982668765318,0.3232282945573366,0.08192982668765318,0.21697365972730848,0.08504574221593363,0.2638558290622853,0.08504574221593363,0.08504574221593363,12.39
1,0.0009960612933065818,27.16890166903207,0.057602403449372425,0.28059255507514286,0.057602403449372425,0.20744694073994954,0.0832311443708561,0.24843159604955603,0.0832311443708561,0.0832311443708561,9.9
2,0.00099115248173898,20.227945951203854,0.054371398274395376,0.26893506141739804,0.054371398274395376,0.1718602987351241,0.06386383286228886,0.21599292821354335,0.06386383286228886,0.06386383286228886,9.91
3,0.0009843072889837512,16.728760903209828,0.0513016399879431,0.26294882856693463,0.0513016399879431,0.1733362728798831,0.07242060281612255,0.20183134630874353,0.07242060281612255,0.06386383286228886,10.01
4,0.0009755527298894294,14.361599259408335,0.05017015325744778,0.2658989545140865,0.05017015325744778,0.16680129611933672,0.06176700051184054,0.21006858944892884,0.06176700051184054,0.06176700051184054,11.33
16,0.0007411359602138069,2.937722257587532,0.04641859551689493,0.25852949437479167,0.04641859551689493,0.1391044192843967,0.05278937557229289,0.17263009183936648,0.05278937557229289,0.04655048946539561,12.07
17,0.0007131767561367538,2.5893935685883585,0.04640143865289075,0.25464335909896074,0.04640143865289075,0.15844571667688864,0.05939706718480146,0.19809729191992018,0.05939706718480146,0.04655048946539561,11.12
18,0.0006843782140659967,2.296900770460352,0.04670952351988627,0.2632308842884258,0.04670952351988627,0.1457585248682234,0.05735698565840721,0.17680307803330597,0.05735698565840721,0.04655048946539561,11.13
19,0.0006548539886902863,2.0391442300768747,0.04629717971689055,0.25914175131597106,0.04629717971689055,0.1758549381185461,0.0767553279797236,0.19819921652475994,0.0767553279797236,0.04655048946539561,11.39
20,0.0006247205986388449,1.8210994666673834,0.0464046179136773,0.2638112526723585,0.0464046179136773,0.12236080169677735,0.04931380268600252,0.14609399459980152,0.04931380268600252,0.04655048946539561,10.91
```

(For the full CSV, attach the file separately.)

### D.5 Training curves figure

Attach `results/figures/training_curves.png` to the handover.

### D.6 Day-7 gate decision

Per Custom Instructions §11 / TRAINING_GUIDE.md §3.2:

| Best val MAE band | Verdict |
|---|---|
| ≤ 0.020 | PASS — target hit |
| 0.020 - 0.025 | PASS — floor met |
| 0.025 - 0.040 | BORDERLINE |
| > 0.040 | FAIL |

**Decision:** **FAIL**

**Rationale:** Best val MAE 0.0466 > hard floor 0.040. Stop normal flow; apply fallback (drop physics loss, retrain pure regressor) per TRAINING_GUIDE §6.

<!-- ====== Manually fill the remaining [FILL AFTER RUN] sections (D.2, D.5, D.7, P1-4, P2-2, P2-3, P2-7, B.3, C-extra) ====== -->

### D.7 Sanity-check observations

(Cross-check from `results/figures/training_curves.png` — see
TRAINING_GUIDE.md §3.3.)

- Train and val loss both monotone decreasing? **[FILL — Yes / No / Mostly]**
- Val MAE crossed the 0.025 line? **[FILL — Yes / No]**
- Val MAE crossed the 0.020 line? **[FILL — Yes / No]**
- Physics loss decreased? **[FILL — Yes / Flat / Increased]**
- Train ≪ val (gap > 2× sustained > 10 epochs)? **[FILL — Yes / No]**
- Any NaN / Inf during training? **[FILL — Yes / No]**
- LR schedule completed full cosine? **[FILL — Yes / Stopped early]**

---

## E. What Chat 2 (T17 mid-checkpoint) needs to do

### E.1 Inputs available

1. **`checkpoints/best.pt`** — contents:
   ```python
   {
       "model": <state_dict>,
       "optimizer": <state_dict>,
       "scheduler": <state_dict or None>,
       "epoch": <int>,
       "val_metrics": {"val_mae": float, "best_val_mae": float, "val_loss_total": float},
       "config": <full merged config dict>,
   }
   ```
   Load with `weights_only=False` (it contains a config dict).

2. **`engine/reference_spectra.npy`** — `(6, 1024)` float32 in canonical
   compound order. Required by `RamanPhysicsAI` constructor.

3. **`data/processed/spectra_full.pt`** + `labels.pt` — preprocessed AA
   data, ready to load.

4. **`data/splits/split_A_composition_ood.json`** — `{"train", "val", "test"}` index lists. T17 must use `test` (not val, not train).

5. **`src/eval/metrics.py`** — exposes `quantification_mae`,
   `identification_accuracy`, `ood_auroc`, `constraint_violation_rate`,
   `reconstruction_cosine_similarity`. Built in Phase A.

### E.2 T17 task contract

Chat 2 should:

1. Build the model from `ck["config"]` and load `ck["model"]` weights.
2. Build a TEST DataLoader from `test` indices in the split JSON.
   No augmentation. Batch size doesn't matter for eval.
3. Forward all test spectra through the model. Collect:
   - `y_pred` (B_total, 6) compositions
   - `s_recon` (B_total, 1024) reconstructions
4. Compute the 5 metrics:
   - `quantification_mae(y_true, y_pred)` → headline number
   - `identification_accuracy(y_true, y_pred, threshold=0.05)` → fraction of samples where dominant compound prediction matches
   - `reconstruction_cosine_similarity(s_input, s_recon)` → returns dict, report `median`
   - `constraint_violation_rate(y_pred)` → fraction of rows that aren't a valid simplex (should be ~0 thanks to softmax)
   - **Skip `ood_auroc`** — needs OOD samples not yet defined; defer to Phase 3 once T19 builds the OOD scorer
5. Write `results/midcheckpoint_report.md` with:
   - Numbers for the 4 applicable metrics
   - Comparison vs Custom Instructions §6 targets/floors
   - Explicit GO / NO-GO recommendation (mirror D.6 here, but on TEST not val)
6. Save predictions for later analysis: `results/midcheckpoint_predictions.npz`
   with `y_true`, `y_pred`, `s_input`, `s_recon`.

### E.3 Expected T17 outputs

After Chat 2 finishes T17:

- `results/midcheckpoint_report.md` (new)
- `results/midcheckpoint_predictions.npz` (new)
- A clear PASS/FAIL recommendation that either
  (a) unlocks Phase 3 (T18 MC Dropout + T19 OOD) for Chat 3, or
  (b) triggers fallback work in Chat 3 (simplification, retrain).

### E.4 Compound order is locked

Do NOT reorder anywhere without coordinating with `engine/reference_spectra.npy`:

```python
CANONICAL_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]
```

`labels.pt` columns, `reference_spectra.npy` rows, model output index,
`bond_mapping.json` compound names — all use this order.

---

## F. Return path: Chat 2 → Chat 3 (Phase 3)

After Chat 2 finishes T17:

| T17 verdict | Chat 3 next action |
|---|---|
| GO (PASS-target or PASS-floor) | Phase 3: T18 (MC Dropout wrapper) + T19 (OOD score). ~8h work. |
| BORDERLINE | Phase 3 still proceeds, but T18 also acts as a sanity check (high MC variance ⇒ model uncertain ⇒ explains low MAE). Document caveats in REPORT.md. |
| FAIL | Phase 3 paused. Fallback work in Chat 3: drop physics loss (`beta_phys: 0`), drop reconstruction module from `forward`, retrain pure regressor. Re-run T17 once that retrain produces a checkpoint. |

---

## G. Quick reproduction recipe

```powershell
# Prerequisites: Phase 1 + Phase 2 files dropped in,
# data/processed/* and engine/reference_spectra.npy built per
# TRAINING_GUIDE.md §0.

# 1. Tests pass (one-time)
pytest tests\test_models.py tests\test_augmentation.py -v

# 2. Smoke (~3 min)
python -m src.training.train --config configs/train_config.yaml --smoke
Remove-Item checkpoints\best.pt, checkpoints\last.pt, results\training_log.csv -ErrorAction SilentlyContinue

# 3. Full training (~30-90 min)
python -m src.training.train --config configs/train_config.yaml

# 4. Inspect curves
python scripts\plot_training_curves.py
start results\figures\training_curves.png

# 5. Verify checkpoint loads
python -c "import torch; ck=torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False); print('epoch:', ck['epoch'], 'val_mae:', ck['val_metrics']['val_mae'])"

# 6. Hand to Chat 2 — paste THIS file (after filling [FILL AFTER RUN] sections),
# attach training_log.csv and training_curves.png.
```

---

*End of Chat 3 (Core Engine) handover. Document version 1.0 (template).
Will be updated to version 1.1 with [FILL AFTER RUN] sections completed
once user runs training locally.*
