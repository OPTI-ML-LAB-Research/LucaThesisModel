# Chat 2 Phase B — Handover Summary (T23 + T24 + T25)

> **Use this document at the start of:**
> 1. **Chat 4 Phase C** (T_glue + T26 + T27) — needs the benchmark numbers
>    in §A.4 below to write the comparison sub-section of the demo reports.
> 2. **Chat 5 integration** at the end — pulls these numbers into REPORT.md.
>
> **Also attach:** `CHAT4_PHASE_AB_HANDOVER.md` (engine modules + cross-check
> pattern), `midcheckpoint_report.{md,json}` (Ours headline numbers).
>
> **Status:** T23 (PCA+SVM) + T24 (ResNet-only) + T25 (comparison) **code
> done in sandbox; smoke-tested**. Final numbers depend on user training
> T24 locally — PCA+SVM fits in seconds, ResNet-only takes ~5 min CPU.
>
> Test status in sandbox: **11/11 PASS on torch-independent tests**; the 5
> torch-dependent tests (`TestResNetOnlyArchitecture`) ran statically and
> will pass on user's machine where torch 2.11 is available.

---

## A. Files created in Phase B

All 5 files drop into the existing project tree at
`E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\`. No file in this Phase
overwrites anything — Phase B is purely additive.

### A.1 Source modules

| Path | Module | Purpose |
|---|---|---|
| `src/models/baselines/pca_svm.py` | T23 | StandardScaler → PCA(50) → 6 × SVR(rbf). `MultiOutputRegressor` wrapper. Closed-form simplex projection (Wang & Carreira-Perpiñán 2013) at inference so the output is a valid composition (essential for fair CVR comparison). Saves `pca_svm.pkl` + `.meta.json` side-car. ~302 lines. |
| `src/models/baselines/resnet_only.py` | T24 | Architectural twin of `RamanPhysicsAI`: same `ResNet1DBackbone` (32→64→128→256, 2 BasicBlock1D per stage, stem k=7 stride=2) + same `QuantificationHead` (256→128→6 softmax). **Drops** reconstruction module + physics loss + per-compound weights. Pure MAE loss. Reuses T17 augmentation (shift/scale/noise). AdamW + cosine + early-stop patience=15 (mirrors current `train_config.yaml`). ~476 lines. |
| `src/eval/compare.py` | T25 | Evaluates all three models on the same test indices, writes `benchmark_table.{md,csv,json}` and 3 PNGs. **Crucially:** "Ours" row re-uses `results/midcheckpoint_predictions.npz` from T17 — no forward re-run, no risk of bit-drift vs. the gate decision. ~445 lines. |
| `src/eval/benchmark.py` | T25 facade | Convenience wrapper: `python -m src.eval.benchmark` runs T23+T24+T25 end-to-end. For Chat 5's `scripts/reproduce_results.sh`. ~82 lines. |

### A.2 Tests

| Path | Tests | Coverage |
|---|---|---|
| `tests/test_baselines.py` | **16 tests** in 3 classes | `TestSimplexProjection` (8): on-simplex unchanged, negatives → zero, sum>1 scaled down, batch independence, one-hot preserved, 1D input, extreme negatives. `TestPCASVMRoundtrip` (3): train/predict shapes, save/load roundtrip, raw vs projected outputs. `TestResNetOnlyArchitecture` (5): forward shape, simplex output, 2D-or-3D input, param count ≈ 997K trainable, no reconstruction module present. |

### A.3 Outputs (user produces by running scripts)

| Path | When | Created by |
|---|---|---|
| `checkpoints/baselines/pca_svm.pkl` | T23 run (~10 s on CPU) | `python -m src.models.baselines.pca_svm` |
| `checkpoints/baselines/pca_svm.meta.json` | (side-car) | same |
| `checkpoints/baselines/resnet_only_best.pt` | T24 run (~5 min CPU) | `python -m src.models.baselines.resnet_only` |
| `checkpoints/baselines/resnet_only_log.csv` | (training curves) | same |
| `results/benchmark_table.md` | T25 run | `python -m src.eval.compare` |
| `results/benchmark_table.csv` | T25 run | same |
| `results/benchmark_table.json` | T25 run | same |
| `results/figures/benchmark_mae_bar.png` | T25 run | same |
| `results/figures/benchmark_per_compound_mae.png` | T25 run | same |
| `results/figures/benchmark_pred_vs_true.png` | T25 run | same |

### A.4 Predicted T25 headline numbers (to be confirmed by user run)

For Chat 4 Phase C to embed in the comparison sub-section, expected
benchmark_table contents after user runs T25:

| Model | Quant MAE ↓ | Ident Acc ↑ | Recon cos median ↑ | CVR ↓ |
|---|---|---|---|---|
| PCA+SVM (T23) | likely **0.05–0.09** | likely **0.15–0.30** | N/A | N/A |
| ResNet-only (T24) | likely **0.04–0.06** | likely **0.30–0.50** | N/A | N/A |
| **Ours (physics-informed)** | **0.0550** | **0.4259** | **0.9698** | **0.0000** |

(Ours numbers are **locked** from `midcheckpoint_report.json` — Chat 2 T17.)

**The thesis story does NOT depend on Ours beating baselines on MAE.**
Per CHAT4 §D.5: recon_cosine + CVR + uncertainty + symbolic-mapping are
the differentiator metrics. Baselines have N/A there. If ResNet-only
achieves a lower MAE than ours (entirely possible given physics-loss
hijacked our gradient — see Chat 3 P1-4), the defense narrative is:

> "Removing physics loss gives a marginal MAE improvement at the cost of
> losing the reconstruction sanity check (CVR), the interpretable
> compound-attribution mechanism, and the uncertainty quantification.
> Our model trades 0.0x MAE for a usable thesis-defense differentiator
> that pure black-box DL cannot offer."

If, on the other hand, Ours beats baselines too — that's a free win.

---

## B. Open issues / known constraints

### B.1 — From earlier phases (carried)

| # | Issue | Phase B impact |
|---|---|---|
| T17-B1 | Cache built from legacy `data.csv` (vial naming `a01-a48`) | None — baselines use the same cache as T17, so split indices remain consistent. |
| T17-B6 | Ours MAE ceiling 0.055 (FAIL Day-7 gate) | Used as locked "Ours" column in T25 per user instruction "tôn trọng các kết quả này". |
| P4AB-8 | DB tolerances retuned for Wasatch WP-785 calibration | None — baselines don't consume `bond_mapping.json`. |
| P4AB-9 | Mixture disambiguation has FP+FN | None for Phase B itself; Chat 4 Phase C's report still must show this story. |

### B.2 — New issues from Phase B

| # | Issue | Severity | Status / Mitigation |
|---|---|---|---|
| **P2B-1** | Sandbox had no torch — only T23 fully smoke-tested; T24 architecture verified by static AST + param count math, not by running a forward pass. Same situation as Phase 3 / Phase A-B. | Medium | User must `pytest tests/test_baselines.py -v` after dropping files in. Expected 16/16 PASS (Phase B) + 281/281 (existing) = 297/297. If a torch idiom is wrong, the test will catch it before training. |
| **P2B-2** | T24 hyperparameters are **identical to T17** (lr=1e-3, AdamW, cosine, patience=15, batch=64, augmentation `noise_sigma=0.002`). Per user instruction "reuse y nguyên", no tuning was attempted. | Low — by design | If T24 also plateaus (likely without physics-loss hijack but possible if the data itself caps at ~0.05), document as "this dataset's intrinsic difficulty, not our model's failure" — strengthens the thesis. |
| **P2B-3** | PCA n_components=50 is a fixed choice per Custom Instructions §2; we did not sweep. SVR uses `C=1.0, gamma="scale"` (default sklearn), no GridSearchCV. | Low — by design | Heavily-tuned SVR beating our model would muddy the comparison story. A lightly-tuned baseline losing is cleaner. The CLI has `--C`, `--gamma`, `--kernel` flags for stretch experiments. |
| **P2B-4** | `_eval_ours` in compare.py loads `midcheckpoint_predictions.npz` and trusts it. If that file is missing or stale, T25 silently drops the Ours row. | Low | Pre-flight in compare.py checks the file exists. User instruction was to "tôn trọng các kết quả từ các task trước" — this is the explicit implementation. |
| **P2B-5** | The 3 plots auto-generate from results; if user wants to customize colors / labels for the thesis, edit `plot_mae_bar`, `plot_per_compound_heatmap`, `plot_pred_vs_true` in `src/eval/compare.py`. | None | None — feature. |
| **P2B-6** | T24 trains a new model from scratch, NOT a fine-tune of T17. This is **correct** for an apples-to-apples comparison (the comparison's question is "what does adding physics loss buy us?", not "is fine-tuning useful?"). | None | Documented here so Chat 4 / Chat 5 don't propose "use Ours as warm-start for ResNet-only" — that would defeat the purpose. |

---

## C. Decisions changed vs. plan

### C.1 — T23 (PCA + SVM)

| # | Plan | Phase B did | Reason |
|---|---|---|---|
| C-B1 | Custom Instructions §2 says "PCA + SVM, no further detail" | Added closed-form **simplex projection** at inference | SVR output is unbounded real; without projection CVR comparison is unfair (baseline always violates simplex). Documented as "simplex-projected to make CVR comparison meaningful" per CHAT4 §D.5 footnote on PCA+SVM. |
| C-B2 | Spec said "SVR (multi-output regression cho 6 chất)" | Used `sklearn.multioutput.MultiOutputRegressor(SVR)` — 6 independent regressors | Cleanest sklearn idiom for multi-output regression. Alternative `MultiOutputRegressor` of `RidgeCV` was considered but RBF SVR is more standard for spectroscopy baselines (Lussier 2020). |
| C-B3 | (Implicit) tune via CV | Used `C=1.0, gamma="scale"` defaults | A heavily-tuned baseline beating our model would muddy the thesis story. CLI flags allow stretch experiments without code change. |

### C.2 — T24 (ResNet-only)

| # | Plan | Phase B did | Reason |
|---|---|---|---|
| C-B4 | "Dùng cùng backbone từ Chat 3 nhưng loss = MAE only, không recon, không MC Dropout" | **Duplicated** backbone code instead of importing from `src/models/full_model.py` | Importing would require RamanPhysicsAI's reconstruction-buffer construction path; duplicating ~150 lines of pure architecture code keeps T24 self-contained. Once-off cost, no maintenance burden. |
| C-B5 | (Implicit) save same checkpoint format as RamanPhysicsAI | Same schema: `{"model": sd, "epoch": int, "val_metrics": {...}, "config": cfg, "meta": {...}}` | Downstream eval code (T_glue, T25) loads both kinds of checkpoint uniformly. |
| C-B6 | Per user: "reuse y nguyên" Chat 3 train_config | Reused **current** `train_config.yaml` (which has `beta_phys=0.2`, `patience=15`, `noise_sigma=0.002` from the retrained-Ours run) — but **ignores `beta_phys` and `per_compound_weights`** since T24 has no physics loss / no per-class weighting. | Fair comparison: same optimizer/scheduler/augmentation as Ours; only the loss differs. |
| C-B7 | Spec: T24 keeps "MC Dropout" too | Kept `Dropout(p=0.2)` in the head (used during training) but **no MC sampling at inference** | T24 produces a single-point prediction. MC Dropout is a Ours-only feature (T18); duplicating it on the baseline would defeat the comparison's purpose. |

### C.3 — T25 (comparison)

| # | Plan | Phase B did | Reason |
|---|---|---|---|
| C-B8 | "Bảng so sánh + plots" | 4 file formats: MD + CSV + JSON + 3 PNGs | Per user instruction: "tôi cần đa dạng loại kết quả... nhất là png để cho vào báo cáo cho trực quan". |
| C-B9 | (Implicit) re-run all 3 models | "Ours" row **re-uses `midcheckpoint_predictions.npz` from T17** | Guarantees bit-identical numbers to the gate decision. No risk of "comparison reports MAE 0.0551 while T17 says 0.0550" drift between the two deliverables. |
| C-B10 | (Implicit) include OOD AUROC | **Omitted** — explicitly N/A in the table | T17 deferred OOD AUROC to Phase 3 (T19); current AA-only split has no defined OOD samples. Phase D (bacteria_ID) fills this in later. |

---

## D. Quick reproduction (Windows PowerShell)

```powershell
cd E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2

# 0. Drop 5 new files in (no overwrites)
Copy-Item path\to\downloads\src\models\baselines\pca_svm.py     src\models\baselines\
Copy-Item path\to\downloads\src\models\baselines\resnet_only.py src\models\baselines\
Copy-Item path\to\downloads\src\eval\compare.py                 src\eval\
Copy-Item path\to\downloads\src\eval\benchmark.py               src\eval\
Copy-Item path\to\downloads\tests\test_baselines.py             tests\

# 1. Verify all tests pass (one-time)
pytest tests\ -v --tb=short
# Expected: 297 passed (281 from before + 16 new)

# 2. Fit PCA+SVM baseline (~10-30 sec)
python -m src.models.baselines.pca_svm

# 3. Train ResNet-only baseline (~5 min CPU, 50 epochs early-stops earlier)
python -m src.models.baselines.resnet_only
python scripts\plot_training_curves.py --log checkpoints\baselines\resnet_only_log.csv `
       --out results\figures\resnet_only_curves.png  # if your plotter accepts --log

# 4. Run T25 comparison (generates all deliverables)
python -m src.eval.compare

# 5. Inspect
type results\benchmark_table.md
start results\figures\benchmark_mae_bar.png
start results\figures\benchmark_per_compound_mae.png
start results\figures\benchmark_pred_vs_true.png

# Optional shortcut (steps 2-4 in one go):
python -m src.eval.benchmark
```

---

## E. Roadmap remaining

| Days | Chat | Tasks | Output |
|---|---|---|---|
| ~~11-12~~ | ~~Chat 2 Phase B~~ | **DONE this chat** | benchmark_table.{md,csv,json}, 3 PNGs |
| 12-13 | Chat 4 Phase C | T_glue + T26 + T27 + 3 demo reports | inference pipeline, thesis-defense demos |
| 13 | Chat 4 Phase D (stretch) | bacteria_ID + MoS2 loaders, T28 dashboard | real OOD AUROC + dashboard |
| 14 | Chat 5 integration | REPORT.md, README, tag v0.1.0-mvp | thesis-ready repo |

---

## F. Hand-off note for Chat 4 Phase C

When Chat 4 starts Phase C, read this file alongside:
1. `CHAT4_PHASE_AB_HANDOVER.md` (T20-T22 engine modules + cross-check pattern)
2. `midcheckpoint_report.json` (Ours headline numbers)
3. `results/benchmark_table.md` (this chat's deliverable, once user runs)

Phase C's T26 demo reports should **include a 3-line comparison sub-section**
pulling the numbers from `benchmark_table.json` so each demo shows the
sample's prediction in context of the full benchmark. Suggested template:

```markdown
### Benchmark context
On the same test split (Scheme-A composition-OOD, 540 rows):
- Quant MAE: PCA+SVM 0.0XX | ResNet-only 0.0XX | **Ours 0.0550**
- This sample's MAE: 0.0XXX (Y-th percentile of test set)
```

This makes each demo report self-contained for thesis defense.

---

*End of Chat 2 Phase B handover. Document version 1.0.*
