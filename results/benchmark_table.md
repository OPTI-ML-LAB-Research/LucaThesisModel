# Benchmark — T25

*Three models, same test split (Scheme-A composition-OOD, 540 rows).*

| Model | Quant MAE ↓ | Ident Acc ↑ | Recon cos median ↑ | CVR ↓ | Inference (ms/sample) |
|---|---|---|---|---|---|
| PCA+SVM | 0.0479 | 0.4981 | N/A | N/A | 0.09 |
| ResNet-only (no physics) | **0.0462** | **0.5056** | N/A | N/A | 2.34 |
| Ours (physics-informed) | 0.0550 | 0.4259 | **0.9698** | **0.0000** | N/A |

Notes:
* Recon cosine and CVR are **only meaningful for our model** — the baselines have no reconstruction module. N/A entries are honest.
* OOD AUROC is omitted from this table; it requires OOD samples not defined for the AA-only test split. Phase D will fill this in.
* Inference time uses the same hardware for all rows (user's CPU). Our-model timing is recorded inside the T17 run, not re-measured here.

## Per-compound MAE (lower = better)

Useful for diagnosing whether the comparison's MAE gap is uniform across compounds, or driven by one hard compound (e.g. Glucosamine).

| Model | Alanine | Asparagine | Aspartic Acid | Glutamic Acid | Histidine | Glucosamine |
|---|---|---|---|---|---|---|
| PCA+SVM | 0.0605 | 0.0605 | 0.0465 | **0.0420** | 0.0329 | 0.0452 |
| ResNet-only (no physics) | **0.0437** | **0.0567** | 0.0671 | 0.0489 | **0.0293** | **0.0318** |
| Ours (physics-informed) | 0.0588 | 0.0709 | **0.0331** | 0.0497 | 0.0515 | 0.0658 |
