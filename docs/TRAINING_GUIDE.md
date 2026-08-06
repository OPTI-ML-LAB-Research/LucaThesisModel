# Training Guide — Phase 2 → Day 7 Mid-Checkpoint

> **Audience:** the project owner (you), running on Windows with the
> Raman-Physics-AI-v2 project tree at `E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\`.
> **Goal:** train `RamanPhysicsAI` end-to-end, get a checkpoint, decide
> whether the Day-7 gate is PASS / BORDERLINE / FAIL, and prepare to hand
> over to Chat 2 for T17 mid-checkpoint evaluation.
>
> **Time budget:** ~15 min smoke + 30-90 min full training (CPU) or
> 5-15 min (GPU). All commands shown in PowerShell.

---

## 0. Pre-flight checklist (run ONCE)

Before anything else, verify the project state. Stop at the first failure
and fix before moving on.

### 0.1 Files in place (Phase 1 + Phase 2 deliverables)

```powershell
cd E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2

# Should print 7 file paths if everything is dropped in correctly.
Get-ChildItem -Recurse `
    src\models\backbone.py, `
    src\models\heads.py, `
    src\models\reconstruction.py, `
    src\models\full_model.py, `
    src\training\losses.py, `
    src\data\augmentation.py, `
    src\training\train.py `
    | Select-Object FullName
```

If any of these are missing, copy them from the Phase 1 / Phase 2 outputs.

### 0.2 Configs

```powershell
Test-Path configs\default.yaml         # must be True (from Groundwork T02)
Test-Path configs\train_config.yaml    # must be True (from Phase 2)
```

### 0.3 Data caches (built by `prepare_data.py` from `AA_Data.csv`)

```powershell
Test-Path data\processed\spectra_full.pt    # (4378, 1024) float32
Test-Path data\processed\labels.pt          # (4378, 6) float32
Test-Path data\processed\wavenumbers.npy    # (1024,) float64
Test-Path data\processed\vial_ids.npy       # (4378,) string
```

If any are False, rebuild:

```powershell
python scripts\prepare_data.py --dataset AA --scheme all
```

### 0.4 Reference spectra (built by `extract_pure_references.py` from ENLIGHTEN exports)

```powershell
Test-Path engine\reference_spectra.npy    # (6, 1024) float32
```

If False, build it:

```powershell
python scripts\extract_pure_references.py `
    --pure-dir data\raw\pure `
    --target-wavenumbers data\processed\wavenumbers.npy `
    --apply-preprocessing
```

After it finishes, **open `results\sanity\pure_spectra.png` and visually
verify** that the Histidine reference shows peaks at ~1003, 1180, 1495,
1575 cm⁻¹ and Glucosamine shows peaks at ~1080, 1100 cm⁻¹. If those
peaks are missing or massively shifted, do NOT proceed to training —
something is wrong with the wavenumber alignment (see CHAT2 handover
Issue #3).

### 0.5 Splits

```powershell
Test-Path data\splits\split_A_composition_ood.json
```

If False, build it (run from project root, this is the snippet from
CHAT2 handover Section F):

```powershell
python -c @"
import numpy as np, torch
from src.data.splits import split_A_vial_level, save_split
labels   = torch.load('data/processed/labels.pt').numpy()
vial_ids = np.load('data/processed/vial_ids.npy', allow_pickle=True).tolist()
save_split(split_A_vial_level(vial_ids, labels, seed=42),
           'data/splits/split_A_composition_ood.json')
"@
```

Expected split sizes after this: `train=3298, val=540, test=540` (per
CHAT2 handover decision C4).

### 0.6 Python environment

```powershell
python -c "import torch, numpy, yaml, pandas, scipy; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
```

If `cuda=True`, training will use GPU automatically (config sets
`device.prefer: auto`). If False, training falls back to CPU (slower
but fine — ~30-60 min for 50 epochs on a modern CPU).

Optional but useful:

```powershell
python -c "import tqdm" 2>$null && echo "tqdm OK (progress bars)"
```

### 0.7 Tests pass

```powershell
pytest tests\test_models.py -v        # 32 tests, expect all PASS
pytest tests\test_augmentation.py -v  # 20 tests, expect all PASS
```

If any test fails, **STOP** — open the failure, fix the underlying
module before training. A failing model test means the trained
checkpoint will be useless.

---

## 1. Smoke test (~3 min) — DO THIS FIRST

Smoke mode runs 2 epochs on a 200-train + 50-val subset. Its only
purpose: confirm the whole pipeline holds together without crashing.
Loss values from smoke mode are **not informative** — don't read into them.

```powershell
python -m src.training.train --config configs/train_config.yaml --smoke
```

### 1.1 What you should see

```
HH:MM:SS train INFO | Loaded base config: configs\default.yaml
HH:MM:SS train INFO | Loaded train config: configs\train_config.yaml
HH:MM:SS train INFO | Device: cpu      (or cuda)
HH:MM:SS train INFO | Loaded 4378 spectra, 6 compounds
HH:MM:SS train INFO | Split sizes: train=3298, val=540, test=540
HH:MM:SS train WARN | SMOKE MODE: subsampled to train=200, val=50
HH:MM:SS train INFO | Augmentation ENABLED: shift+/-6px, scale=(0.9, 1.1), noise_sigma=0.005
HH:MM:SS train INFO | Model params: backbone=963,488 head=33,158 recon=6 total=996,652
HH:MM:SS train INFO | Starting training: 2 epochs, early-stop patience=8
HH:MM:SS train INFO | epoch 000 | lr=1.00e-03 | train mae=0.NNNN loss=0.NNNN | val mae=0.NNNN loss=0.NNNN | best=0.NNNN * | NN.Ns
HH:MM:SS train INFO | epoch 001 | lr=N.NNe-04 | train mae=0.NNNN loss=0.NNNN | val mae=0.NNNN loss=0.NNNN | best=0.NNNN * | NN.Ns
HH:MM:SS train INFO | Done. Best val MAE = 0.NNNN
HH:MM:SS train INFO | Best checkpoint:    checkpoints/best.pt
HH:MM:SS train INFO | Training log CSV:   results/training_log.csv
```

### 1.2 Smoke pass criteria

- ✅ Script exits with code 0 (no Python traceback at the end)
- ✅ Both epochs printed; epoch 001 finished
- ✅ `checkpoints/best.pt` and `checkpoints/last.pt` both exist
- ✅ `results/training_log.csv` has 2 data rows (plus header)
- ✅ `train mae` and `val mae` are finite numbers (not `nan` or `inf`)
- ✅ Param counts match: `backbone=963,488 head=33,158 recon=6`

If any of these fail, do **not** proceed to full training. See
Section 4 (Troubleshooting) below.

### 1.3 Clean up smoke artifacts before full training

Smoke checkpoints would otherwise contaminate the real run via
`--resume`. Wipe them:

```powershell
Remove-Item checkpoints\best.pt, checkpoints\last.pt -ErrorAction SilentlyContinue
Remove-Item results\training_log.csv -ErrorAction SilentlyContinue
```

---

## 2. Full training run (~30-90 min CPU / ~5-15 min GPU)

```powershell
python -m src.training.train --config configs/train_config.yaml
```

Or, if you want to keep the terminal output saved:

```powershell
python -m src.training.train --config configs/train_config.yaml 2>&1 | Tee-Object -FilePath results\training_console.log
```

### 2.1 What to watch in real time

Open a **second terminal** alongside the training one and monitor the
CSV as it grows (it flushes after every epoch):

```powershell
Get-Content results\training_log.csv -Wait -Tail 5
```

You can also re-render the curves PNG at any time without stopping
training (the CSV is read-only in the script):

```powershell
python scripts\plot_training_curves.py
start results\figures\training_curves.png
```

### 2.2 Healthy progression looks like this

| Epoch | train MAE | val MAE | Notes |
|---|---|---|---|
| 0    | 0.18-0.25 | 0.16-0.22 | Random softmax start (~uniform → ~MAE 0.166) |
| 1-5  | drops to 0.05-0.10 | drops to 0.05-0.10 | Backbone learning |
| 5-15 | 0.03-0.05 | 0.03-0.05 | Refining; train MAE may dip below val |
| 15-30 | 0.015-0.03 | 0.018-0.030 | Approaching target |
| 30-50 | 0.010-0.020 | 0.018-0.025 | Plateau / fine-tuning |

**Realistic expectation:** val MAE in **[0.018, 0.025]** by epoch ~30.
Anything ≤ 0.020 is "target" per Custom Instructions §6; ≤ 0.025 is
"floor".

### 2.3 Sanity checks DURING training

- **Loss components** in the CSV: `train_loss_quant` should drop fastest
  in the first 5 epochs; `train_loss_physics` should drop more slowly
  but steadily. If `train_loss_physics` STAYS flat or increases while
  `train_loss_quant` drops, the model is "cheating" — predicting
  composition without actually explaining the spectrum. See Section 4.7.

- **Gradient clipping**: with `grad_clip_norm=1.0` and combined loss
  scale ~1, clipping should rarely fire. The script doesn't log clip
  events explicitly; if loss curves are very flat at epoch 1-2, raise
  `grad_clip_norm` to 5.0 or 0 (off) in `train_config.yaml`.

- **Early stopping**: with `patience=8` and `min_delta=1e-5`, training
  typically stops between epoch 30 and 50. Stopping before epoch 15
  signals trouble (try lowering `learning_rate` or disabling
  augmentation temporarily — see 4.6).

---

## 3. Did it work? — Day 7 gate decision

After training completes (or early-stops), run these checks before
deciding GO / NO-GO.

### 3.1 Read the final numbers

```powershell
# Last 10 epochs of training:
Get-Content results\training_log.csv -Tail 10

# Single number: best val MAE achieved
python -c @"
import csv
with open('results/training_log.csv') as f:
    rows = list(csv.DictReader(f))
best = min(float(r['val_mae']) for r in rows if r['val_mae'])
final_epoch = int(rows[-1]['epoch'])
print(f'Best val MAE: {best:.4f}')
print(f'Final epoch: {final_epoch}')
print(f'Total epochs trained: {len(rows)}')
"@
```

### 3.2 Pass / Fail matrix

| Best val MAE | Status | Action |
|---|---|---|
| ≤ 0.020 | **PASS (target hit)** | Proceed to Phase 3 (MC Dropout + OOD); send handover to Chat 2 for T17 confirmation |
| 0.020 - 0.025 | **PASS (floor met)** | Proceed; document MAE as a known limitation in REPORT.md |
| 0.025 - 0.040 | **BORDERLINE** | Try one of: (a) train 30 more epochs from `last.pt` (`--resume`); (b) lower `lr` to 5e-4; (c) disable augmentation. Re-run, then re-evaluate. Budget: 1 day max |
| > 0.040 | **FAIL — Day 7 gate not passed** | Stop normal flow. Apply fallback: drop physics loss (`beta_phys: 0`), drop reconstruction module from forward, retrain as plain regressor. If THAT also fails, the bug is in data/preprocessing/labels — debug there before more training |

The matrix maps to Custom Instructions §11 ("Val MAE ≤ 0.040 hard floor").

### 3.3 Cross-check sanity

Look at `results/figures/training_curves.png`. Healthy curves:

- **Total loss panel**: train and val both decrease monotonically;
  small gap (val ≥ train by ~10-20%); no oscillation
- **MAE panel**: val MAE crosses the orange (0.025) and ideally the
  green (0.020) reference line; best-val-mae line is staircase
  monotonic-decreasing
- **Loss components panel**: physics loss decreases (slowly is OK);
  if it goes UP for >5 consecutive epochs, that's a warning
- **LR panel**: smooth cosine curve from 1e-3 down to ~1e-6

Unhealthy patterns to flag:

- **Spike + recover**: loss jumps 10× then drops back → grad clip too loose; tighten to 0.5
- **Train ≪ val (gap > 2×)**: overfitting; increase dropout to 0.3 or augmentation probabilities to 0.7
- **Val < train**: usually fine for early epochs (dropout active in train, off in val); flag if it persists past epoch 10
- **Val MAE flat from epoch 1**: optimizer not stepping; double-check `lr > 0`, no `requires_grad=False` on backbone
- **NaN at epoch 1**: physics loss exploded; lower `beta_phys: 0.5 → 0.05` and retry

### 3.4 Verify the checkpoint loads cleanly

Before sending handover to Chat 2, confirm `best.pt` is loadable:

```powershell
python -c @"
import torch
ck = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
print('keys:', sorted(ck.keys()))
print('epoch:', ck['epoch'])
print('val_metrics:', ck['val_metrics'])
print('model state_dict size:', len(ck['model']))
"@
```

Expected output:
```
keys: ['config', 'epoch', 'model', 'optimizer', 'scheduler', 'val_metrics']
epoch: <some int between 10 and 50>
val_metrics: {'val_mae': 0.0NNN, 'best_val_mae': 0.0NNN, 'val_loss_total': 0.0NNN}
model state_dict size: <about 100-150>
```

If this fails, training succeeded but checkpointing didn't — re-run
training (the bug is in `save_checkpoint`).

---

## 4. Troubleshooting

### 4.1 `FileNotFoundError: data/processed/spectra_full.pt`

Cache hasn't been built (or was built then deleted). Run Section 0.3.

### 4.2 `FileNotFoundError: engine/reference_spectra.npy`

Run Section 0.4. The training script fails loud here intentionally —
without real refs, the reconstruction module is meaningless.

### 4.3 `ValueError: Spectrum length mismatch`

Cache was built with a different `spectrum_length`. Either rebuild
cache or update `data.spectrum_length` in `configs/default.yaml` to
match the cache. The AA dataset is 1024 pixels.

### 4.4 `RuntimeError: CUDA out of memory`

Lower `data.batch_size` in `train_config.yaml` to 32 or 16. With this
model size (~1M params), batch 64 should fit on any GPU with ≥4GB.

### 4.5 Training is too slow (CPU)

Options, in order of impact:
1. Use GPU if available (10×+ speedup)
2. Lower `data.batch_size` to 32 — no, this makes it SLOWER on CPU
   because of fewer optimised matmul calls. Counter-intuitive but true.
3. Lower `epochs` to 30 in `train_config.yaml`. Cosine annealing
   adapts; 30 epochs often suffice.
4. Disable augmentation (`augmentation.enabled: false`) — gives ~10%
   speedup but hurts generalisation; only for debugging.
5. On Linux/WSL, set `data.num_workers: 4` for ~1.5× speedup.

### 4.6 Loss not decreasing (val MAE stuck near 0.166)

0.166 ≈ 1/6, which is what MAE looks like when softmax outputs ~uniform
(mean over 6 compounds of |true - 1/6|). Causes:

- `lr` too high → loss exploded once and softmax saturated. Lower
  `learning_rate` to 5e-4 or 1e-4
- `requires_grad=False` accidentally set on backbone (verify with
  Section 3.4, count of model state_dict keys should be ~100+)
- `gamma_reg` too high → L2 dominates and pushes everything toward 0.
  Lower `loss.gamma_reg` to 0.001 or 0.0

### 4.7 Physics loss high while quantification low (model "cheating")

This is the failure mode the architecture is designed to avoid.
Diagnostic:

```powershell
python -c @"
import csv
with open('results/training_log.csv') as f:
    rows = list(csv.DictReader(f))
last = rows[-1]
print(f'Final epoch: {last[\"epoch\"]}')
print(f'  train quant   = {float(last[\"train_loss_quant\"]):.4f}')
print(f'  train physics = {float(last[\"train_loss_physics\"]):.4f}')
print(f'  ratio physics/quant = {float(last[\"train_loss_physics\"])/max(float(last[\"train_loss_quant\"]),1e-6):.2f}')
"@
```

If `physics/quant > 5`, the model isn't using the reconstruction signal.
Options:

- Increase `loss.beta_phys` from 0.5 to 1.0 or 2.0 (more weight on physics)
- Increase `loss.lambda_cosine` from 0.3 to 1.0 (more weight on shape match
  inside physics loss; the MSE term is on absolute scale and may be tiny)
- Verify `engine/reference_spectra.npy` actually contains the right
  compounds in the right order (re-open `results/sanity/pure_spectra.png`)

### 4.8 NaN / Inf appearing

Most likely cause: gradient explosion in physics loss when reconstruction
is far off. Two fixes:

```yaml
# In train_config.yaml
training:
  grad_clip_norm: 0.5      # tighter
loss:
  beta_phys: 0.05          # warm-start with low physics weight
```

After 10 epochs with low `beta_phys`, you can manually `--resume` and
ramp up the weight. (A proper warm-up schedule is Phase 3 work.)

### 4.9 `ImportError: No module named 'src'`

Run from project root, not from inside `src/`:

```powershell
cd E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2     # NOT cd src
python -m src.training.train --config configs/train_config.yaml
```

If your venv doesn't have project root on `sys.path`, also add an
empty `src/__init__.py` (might already exist) and run `pip install -e .`
once to register the package.

### 4.10 Resume from a partial run

Crashed at epoch 23, want to continue?

```powershell
python -m src.training.train --config configs/train_config.yaml --resume
```

The script picks up at epoch 24 from `checkpoints/last.pt`, restores
optimizer + scheduler state, and APPENDS to the existing training log
CSV. If you want a fresh log, delete `results/training_log.csv` first
(or rename it to keep history).

---

## 5. After training — assemble handover for Chat 2

Once Day 7 gate passes (or you have a clear FAIL diagnosis), you'll
edit `CHAT3_CORE_ENGINE_HANDOVER.md` with the actual numbers from your
run, then paste the result into Chat 2 to trigger T17. The handover
template has explicit `[FILL AFTER RUN]` markers showing what to
replace.

Quick collection:

```powershell
# 1. Best val MAE
python -c "import csv; rows=list(csv.DictReader(open('results/training_log.csv'))); print(min(float(r['val_mae']) for r in rows))"

# 2. Best epoch
python -c @"
import csv
rows = list(csv.DictReader(open('results/training_log.csv')))
best_idx, best_row = min(enumerate(rows), key=lambda kv: float(kv[1]['val_mae']))
print(f'Best epoch: {best_row[\"epoch\"]}, val_mae={best_row[\"val_mae\"]}')
"@

# 3. Final epoch + early-stop reason
python -c @"
import csv
rows = list(csv.DictReader(open('results/training_log.csv')))
print(f'Final epoch: {rows[-1][\"epoch\"]} of {len(rows)} total epochs trained')
print(f'Final val_mae: {rows[-1][\"val_mae\"]}')
"@

# 4. Files to attach when handing over to Chat 2
Get-ChildItem checkpoints\best.pt, results\training_log.csv, results\figures\training_curves.png
```

You'll attach (or paste contents of):
- `results/training_log.csv` (whole file — small)
- `results/figures/training_curves.png` (Chat 2 will look at this)
- The completed `CHAT3_CORE_ENGINE_HANDOVER.md`

Chat 2 doesn't need `best.pt` directly — it will load it from disk
when running T17 in your local environment.

---

## 6. Decision tree summary

```
SMOKE PASS?
   |
   No -> fix per Section 4, retry smoke (don't waste time on full run)
   |
   Yes
   |
   v
FULL TRAINING RUN
   |
   Best val MAE ?
   |
   <= 0.020 ----> PASS-target  -> Phase 3 (MC Dropout + OOD), Chat 2 T17
   <= 0.025 ----> PASS-floor   -> same; note limitation
   <= 0.040 ----> BORDERLINE   -> 1 day to retune (Sec 3.2), then decide
   >  0.040 ----> FAIL gate    -> Stop, simplify (drop physics loss), debug

Whatever the outcome:
- Save training_log.csv + training_curves.png + checkpoints/best.pt
- Fill in CHAT3_CORE_ENGINE_HANDOVER.md
- Hand to Chat 2 for T17 (gives the formal mid-checkpoint write-up)
```

---

*Generated end of Phase 2. Update this guide with anything you learn
during the actual run.*
