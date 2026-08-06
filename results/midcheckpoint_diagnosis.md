# T17 Failure Diagnosis

N_test = 540, N_train = 3298, K = 6

## D1 — Prediction distribution

| Compound | mean | std | min | max |
|---|---|---|---|---|
| Alanine | 0.227 | 0.055 | 0.052 | 0.404 |
| Asparagine | 0.178 | 0.097 | 0.026 | 0.651 |
| Aspartic Acid | 0.173 | 0.060 | 0.031 | 0.298 |
| Glutamic Acid | 0.119 | 0.078 | 0.007 | 0.326 |
| Histidine | 0.184 | 0.113 | 0.056 | 0.572 |
| Glucosamine | 0.119 | 0.029 | 0.032 | 0.221 |

**Verdict:** ⚠ LOW VARIANCE — model is hedging toward a single point

## D2 — Marginal baseline comparison

- mean(y_train) per compound: ['0.172', '0.169', '0.184', '0.159', '0.179', '0.137']
- **Baseline MAE** (predict mean): **0.0716**
- **Model MAE**                 : **0.0550**
- Avg L2 distance of model preds to baseline: 0.1832

**Verdict:** ✅ Model beats marginal baseline; some sample-level signal exists

## D3 — Per-compound MAE

| Compound | MAE | y_true mean | y_pred mean |
|---|---|---|---|
| Alanine | 0.0588 | 0.251 | 0.227 |
| Asparagine | 0.0709 | 0.156 | 0.178 |
| Aspartic Acid | 0.0331 | 0.161 | 0.173 |
| Glutamic Acid | 0.0497 | 0.157 | 0.119 |
| Histidine | 0.0515 | 0.141 | 0.184 |
| Glucosamine | 0.0658 | 0.135 | 0.119 |

Worst: **Asparagine**, best: **Aspartic Acid**

## D4 — Reconstruction `scale` parameter

- key: `reconstruction.scale`
- shape: (6,)
- per-compound: ['0.770', '0.817', '0.805', '0.793', '1.040', '0.557']
- summary: mean=0.797, std=0.140, min=0.557, max=1.040

**Verdict:** ✅ Scale stable (max |scale| = 1.04, near init 1.0). Recon module is NOT cheating.

## D5 — Per-compound Pearson correlation (y_true vs y_pred)

Pearson r tells whether the model has learned anything per-compound. r ≈ 0 → no signal (model is decoupled from input).

| Compound | r | interpretation |
|---|---|---|
| Alanine | +0.589 | moderate |
| Asparagine | +0.532 | moderate |
| Aspartic Acid | +0.870 | strong signal |
| Glutamic Acid | +0.706 | strong signal |
| Histidine | +0.886 | strong signal |
| Glucosamine | -0.237 | weak signal |

**Verdict:** ✅ SIGNAL EXISTS — avg |r| = 0.637. Model differentiates samples.

## Synthesis

- **mode_collapse**: OK
- **marginal_baseline**: OK
- **scale_inflated**: OK
- **no_signal**: OK

**No single dominant failure mode** — diagnostic flags mixed. Review per-compound MAE (D3) and worst-compound spectrum to identify specific weakness.
