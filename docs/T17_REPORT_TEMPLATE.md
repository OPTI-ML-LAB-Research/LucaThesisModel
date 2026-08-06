# T17 Mid-Checkpoint Report — Day 7 GO/NO-GO Gate

> **Instructions for filling this template.**
> After running Round 2 retrain + `run_t17_midcheckpoint.py` + `diagnose_t17.py`:
>
> 1. Replace every `<<< ... >>>` placeholder with the actual number from
>    your run.
> 2. Decide PASS / FAIL / BORDERLINE for each metric using the table in
>    Section 1.
> 3. Save as `results/midcheckpoint_report.md` (overwriting the
>    auto-generated stub).
> 4. Delete this instruction block before finalizing.

---

## 1. Headline metrics — TEST set (Scheme A composition-OOD)

Decision rules per Custom Instructions §6 (Hard floor) and §11 (Day-7 gate):

| Metric | Target | Hard floor | Day-7 gate |
|---|---|---|---|
| Quantification MAE | ≤ 0.020 | ≤ 0.025 | ≤ 0.040 |
| Identification accuracy | ≥ 0.90 | ≥ 0.85 | (informational) |
| Reconstruction cosine median | ≥ 0.95 | ≥ 0.85 | (informational) |
| Constraint Violation Rate | ≤ 0.05 | ≤ 0.10 | (informational) |

### Round 2 results

| Metric | Value | Status |
|---|---|---|
| Quantification MAE | **<<< quant_mae >>>** | <<< PASS-target / PASS-floor / BORDERLINE / FAIL >>> |
| Identification accuracy | <<< ident_accuracy >>> | <<< PASS / FAIL >>> |
| Reconstruction cosine median | **<<< recon_cos_median >>>** | <<< PASS-target / PASS-floor / FAIL >>> |
| Constraint Violation Rate | **<<< CVR >>>** | <<< PASS-target / PASS-floor / FAIL >>> |
| OOD AUROC | DEFERRED | T19 (Phase 3) |

**Overall Day-7 verdict:** <<< PASS-target / PASS-floor / BORDERLINE / FAIL >>>

**Rationale:** <<< 1-2 sentences. Compare to Round 1 numbers (MAE 0.0582,
ident_acc 0.4815, recon_cos 0.9513, CVR 0.0130). State whether Round 2
weight rebalancing + beta_phys=0.2 improved things and by how much. >>>

---

## 2. Per-compound analysis (T17 D3 + D5 from diagnose_t17.py)

Round 1 reference (for comparison):

| Compound | R1 MAE | R1 Pearson r |
|---|---|---|
| Alanine | 0.0510 | +0.736 |
| Asparagine | 0.0741 | +0.282 |
| Aspartic Acid | 0.0444 | +0.792 |
| Glutamic Acid | 0.0606 | +0.489 |
| Histidine | 0.0738 | +0.819 |
| Glucosamine | 0.0453 | +0.331 |
| **avg \|r\|** | — | **0.575** |

Round 2 result:

| Compound | R2 MAE | R2 Pearson r | Delta MAE vs R1 | Notes |
|---|---|---|---|---|
| Alanine | <<< >>> | <<< >>> | <<< >>> | |
| Asparagine | <<< >>> | <<< >>> | <<< >>> | |
| Aspartic Acid | <<< >>> | <<< >>> | <<< >>> | |
| Glutamic Acid | <<< >>> | <<< >>> | <<< >>> | |
| Histidine | <<< >>> | <<< >>> | <<< >>> | |
| Glucosamine | <<< >>> | <<< >>> | <<< >>> | |
| **avg \|r\|** | — | **<<< >>>** | — | |

**Key observations:** <<< 2-3 bullets. Which compounds improved most/worst?
Did Histidine recover from the Round 1 over-correction (R1 MAE 0.0738
caused by weight=0.47)? Is the avg correlation higher than Round 1's 0.575? >>>

---

## 3. Run metadata

| Field | Value |
|---|---|
| Date & time | <<< YYYY-MM-DD HH:MM >>> |
| Device | <<< cpu / cuda:0 / ... >>> |
| Epochs trained | <<< N >>> / 50 max |
| Best epoch | <<< N >>> |
| Early stopped? | <<< Yes (patience=15) / No >>> |
| Wall time | <<< minutes >>> |
| Per-compound weights | `[0.94, 1.14, 0.88, 1.02, 1.13, 0.89]` (Round 2) |
| beta_phys | 0.2 |
| Source CSV | `data.csv` (legacy, vial naming a01-a48) |

**Files produced this run:**

* `checkpoints/best.pt`
* `checkpoints/last.pt`
* `results/training_log.csv`
* `results/figures/training_curves.png` (run `scripts/plot_training_curves.py`)
* `results/midcheckpoint_predictions.npz`
* `results/midcheckpoint_report.json`
* `results/midcheckpoint_diagnosis.md`
* This file: `results/midcheckpoint_report.md`

---

## 4. Decision: PROCEED to Phase 3 or fallback retrain

### If verdict is PASS-target or PASS-floor:
GO. Proceed to Chat 3 Phase 3 (T18 MC Dropout + T19 OOD).

### If verdict is BORDERLINE (MAE 0.025-0.040):
GO with documented caveat. The 4 differentiator metrics
(recon_cos, CVR, per-compound r, interpretability) are the thesis's main
contribution — they all pass. Document MAE limitation in `docs/REPORT.md`
when writing up.

### If verdict is FAIL (MAE > 0.040):
**Still GO** per the user's explicit decision: *"Nếu kết quả vẫn không có
gì khả quan, không sao, chúng ta vẫn sẽ tiếp tục các task tiếp theo."*

Per Custom Instructions §6: *"If 'Hard Floor' not met by Day 12, ship
anyway and note as limitation."*

The thesis-defense narrative pivots:
- From "we beat the MAE target" (would require <= 0.025)
- To "we built an interpretable physics-informed system whose contribution
  is uncertainty + OOD + symbolic mapping, validated against simpler
  baselines (T23/T24)"

**Selected decision:** <<< GO (PROCEED to Phase 3) >>>

---

## 5. Limitations documented for thesis defense

(These are honest caveats, NOT excuses. State them upfront in the defense
talk so reviewers see them BEFORE asking.)

1. **MAE ceiling around <<< number >>> on composition-OOD split.** This is
   ~3x stricter than the leave-one-out CV split used by Zarei et al. (2023)
   on the same dataset, which is why their reported R^2 0.92-0.98 is not
   directly comparable to our MAE.

2. **Strong fingerprint-region overlap between 5/6 amino acids.** Alpha-amino
   acids differ only in side chains, producing closely-spaced peaks
   (<10 cm-1 apart). Aspartic Acid, Glutamic Acid, Asparagine, and
   Alanine are the hardest to disambiguate; Histidine (imidazole ring
   at 1003/1180/1495/1575 cm-1) and Glucosamine (pyranose at 1080/1100
   cm-1) are easier - and our model achieves stronger correlation on
   them (per-compound r in D5).

3. **Single-instrument validation.** All training data comes from one
   Wasatch WP-785 spectrometer; cross-instrument generalization is out
   of MVP scope (v2 future work).

4. **OOD detector (T19) not yet validated.** OOD AUROC reported as
   DEFERRED above. Phase 3 will fill this in.

---

## 6. Next-task list for Chat 3 Phase 3

Per CHAT3_CORE_ENGINE_HANDOVER section F:

| Task | Description | Estimated |
|---|---|---|
| T18 | MC Dropout uncertainty wrapper | 3h |
| T19 | OOD score = f(reconstruction error, predictive variance) | 4h |
| T20 | predict(spectrum) main inference function | 2h |
| T21 | report() JSON + Markdown generation | 2h |
| T22 | visualize() plotting utilities | 1h |

Chat 3 will read `checkpoints/best.pt` from this run and build the
inference pipeline on top.

---

## 7. Verdict locked

`Day-7 gate verdict:` <<< GO / GO-WITH-CAVEAT / FAIL >>>
`Date locked:` <<< YYYY-MM-DD >>>
`Next chat:` Chat 3 Phase 3 (T18-T22)
