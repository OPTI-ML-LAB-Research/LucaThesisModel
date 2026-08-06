# T17 — Mid-Checkpoint Report

*Generated: 2026-05-12T21:57:51*

## ⚠ Pre-flight warnings

- Config has `data_raw_csv = 'data/raw/data.csv'` (legacy). If `spectra_full.pt` was rebuilt from `AA_Data.csv` since training, row order may differ and T17 will silently use wrong indices. Verify by checking vial_ids.npy naming (a01-a48 vs aa01-aa48).

## Inputs

- **Checkpoint:** epoch 12, best_val_mae 0.05227530984966843
- **Split:** `data\splits\split_A_composition_ood.json` (scheme `A_vial_level`, seed `42`)
- **Test rows evaluated:** 540
- **Device:** cpu

## Metrics on TEST set

| Metric | Value | Target | Floor | Band | Note |
|---|---|---|---|---|---|
| Quantification MAE | **0.0550** | 0.02 | 0.025 | **FAIL** | 0.0550 > floor 0.025 |
| Identification Accuracy | **0.4259** | 0.9 | 0.85 | **FAIL** | 0.4259 < floor 0.85 |
| Reconstruction cosine (median) | **0.9698** | 0.95 | 0.85 | **PASS-target** | 0.9698 ≥ target 0.95 |
| Constraint Violation Rate (cos < 0.85) | **0.0000** | 0.05 | 0.1 | **PASS-target** | 0.0000 ≤ target 0.05 |

Note: `ood_auroc` is skipped — needs OOD samples not yet defined. Evaluated in Phase 3 after T19.

Reconstruction distribution (additional diagnostics):

- Mean: **0.9660**, Median: **0.9698**
- Percentiles: p05=0.9413, p25=0.9600, p75=0.9753, p95=0.9806

## Overall verdict

### **FAIL** ❌

Headline quant_mae missed the floor. **Phase 3 paused.** Apply fallback retrain per CHAT3 §F.3:

```yaml
# configs/train_config.yaml overrides
loss:
  beta_phys: 0.2           # was 0.5
  alpha_quant: 1.0
training:
  early_stopping_patience: 15  # was 8
data:
  augmentation:
    gaussian_noise_sigma: 0.002  # was 0.005
```

After retrain produces new `best.pt`, re-run this script.

## Files written

- This report
- `results\midcheckpoint_predictions.npz` — `y_true`, `y_pred`, `s_input`, `s_recon`
