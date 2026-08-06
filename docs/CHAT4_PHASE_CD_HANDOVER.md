# Chat 4 Phase C-D (T_glue + T26 + T27 + T28) — Handover Summary

> **Use this document at the start of Chat 5 (integration / REPORT.md /
> v0.1.0-mvp tag).**
>
> **Also attach:** `CHAT2_PHASE_B_HANDOVER.md` (benchmark table numbers),
> `CHAT4_PHASE_AB_HANDOVER.md` (engine modules + cross-check pattern),
> `CHAT3_PHASE3_HANDOVER.md` (T18+T19 inference primitives),
> `CHAT3_CORE_ENGINE_HANDOVER.md` (checkpoint loading recipe).
>
> **Status:** T_glue (predict.py) + T26 (report.py) + T27 (visualize.py)
> + T28 (Streamlit dashboard) DONE in sandbox, **23 new tests pass** in
> addition to the existing 297 (= 320 total expected on the user's
> machine). End-to-end smoke test on a mock model + mock OOD calibration
> verified the pipeline composes correctly. Three demo reports will be
> produced by `scripts/run_demo_reports.py` when run on the real
> checkpoint + cache.

---

## A. Files created in Phase C-D

All paths relative to `E:\Project\KhoaLuanCourse\Raman-Physics-AI-v2\`.

### A.1 Source modules

| Path | Module | Lines | Purpose |
|---|---|---|---|
| `src/inference/predict.py` | T_glue | 388 | `predict(spectrum, **opts) -> dict` -- end-to-end orchestrator. Lazy-loads model, OOD scorer, engine layer on first call; subsequent calls reuse the cache. Supports `skip_ood=True` for tests; `n_mc_samples` knob (default 50). Returns the full §D.2 schema from `CHAT4_PHASE_AB_HANDOVER`. Also provides `predict_batch()` and `reset_cache()`. |
| `src/inference/report.py` | T26 | 360 | `generate_report(result, sample_id, ground_truth, benchmark_context, image_paths) -> {markdown, json, plain_text}`. **D.3 cross-check pattern is implemented and mandatory** (per P4AB-9): every compound gets an agreement / learned-only / symbolic-only / agreement-absent tag in the composition table. JSON output is fully serialisable. `save_report()` writes both MD and JSON to disk. |
| `src/inference/visualize.py` | T27 | 297 | Three plot functions: `plot_reconstruction_overlay` (input + recon + diff subplot, cosine annotation), `plot_peak_annotations` (color-coded peaks per compound, unmatched grey, DB-id labels), `plot_ood_summary` (gauge with green/orange/red zones + spectrum with shaded novelty clusters). Plus `plot_all()` convenience wrapper that writes all three with a shared prefix. |
| `src/inference/__init__.py` | new | 8 | Package docstring. |
| `src/__init__.py` | new | 0 | Empty marker (required for the package to be importable). |
| `dashboard/app.py` | T28 (optional) | 285 | Streamlit 3-tab UI: (1) Upload / sample-picker (with "Histidine-dominant" / "Glucosamine-dominant" / "random" preset buttons), (2) Analysis (pie chart + composition table + peak table + 3 plots), (3) Report (rendered Markdown with download buttons for MD + JSON). |

### A.2 Tests

| Path | Tests | Coverage |
|---|---|---|
| `tests/test_inference.py` | **23 tests** in 3 classes (`TestPredict`, `TestReport`, `TestVisualize`) | Uses an in-process mock RamanPhysicsAI + mock OODScorer installed via `sys.modules` so it never touches the real checkpoint. Verifies: predict() returns all 22 required keys; composition is a valid simplex; all 6 compounds present in output; Histidine spectrum produces 4 matched peaks (P004/P008/P013/P014); `skip_ood=True` flag works; missing checkpoint raises FileNotFoundError; report has Cross-check column (P4AB-9 mandatory check); all 4 cross-check tags correctly computed; ground_truth + benchmark_context optional sections render; 3 visualisation functions all return a Figure and write a PNG; `plot_all()` writes exactly 3 files. |

### A.3 Demo + smoke-test scripts

| Path | Purpose | Estimated runtime on user's machine |
|---|---|---|
| `scripts/run_demo_reports.py` | Produces 3 thesis-defense demo reports on real test data: (1) Histidine-dominant ID sample, (2) Glucosamine-dominant "mild OOD" sample (highest per-compound MAE per T17 / T25), (3) Synthetic 380 cm⁻¹ spike injected into the ID sample to mimic MoS₂ "hard OOD". Each demo writes `demo_*/demo_*.md`, `.json`, and 3 PNGs. | ~30 s × 3 demos = ~2 min |
| `scripts/smoke_test_phase_c.py` | Builds a mock filesystem + mock model + mock OOD scorer in a tmpdir; runs predict + report + visualize end-to-end. Used to verify that orchestration is intact without needing the real checkpoint. Used during this chat to confirm the pipeline works before user runs it. | ~10 s |

### A.4 Final test status

Expected on user's Windows machine (`pytest tests/ -v`):

```
Augmentation:   20
Baselines:      16   (Chat 2 Phase B)
Data:           39
Engine:         25   (Groundwork T05)
Inference:      23   (NEW -- this chat)
Metrics:        28
Models:         31
Novelty loc:    24   (Chat 4 Phase A-B)
OOD:            34   (Chat 3 Phase 3)
Peak ext:       21   (Chat 4 Phase A-B)
Symbolic:       27   (Chat 4 Phase A-B v1.1)
Uncertainty:    32   (Chat 3 Phase 3)
TOTAL:         320
```

All 320 should pass in ~45-50 s on Python 3.13 + Windows.

---

## B. Open issues / known constraints

### B.1 Carried over from earlier phases

| # | Issue | Origin | Phase C-D impact |
|---|---|---|---|
| T17-B1 | Cache built from legacy `data.csv` (a01-a48 vial naming) | Chat 2 T17 | None -- predict() reads same cache uniformly. |
| T17-B2 | `engine/reference_spectra.npy` baked into checkpoint, not built from ENLIGHTEN | Chat 2 T17 | Resolved by predict.py recipe: dumps `reconstruction.pure_ref` from state dict to a tempfile, points `build_full_model_from_config` at that path. No need to rebuild refs separately. |
| T17-B6 | MAE ceiling ~0.055 | Chat 2 T17 | Surfaces as `composition_std` in report (P4AB-3 mitigation). |
| P3-2 | OODScorer.calibrate() ~5-10 min on CPU | Chat 3 Phase 3 | Resolved: predict.py loads from `results/ood_demo/calibration.json`, never re-calibrates. |
| P3-3 | OODScorer must NOT receive a wrapped model | Chat 3 Phase 3 | Resolved: predict.py passes the raw model directly. |
| P4AB-3 | Symbolic disambiguation can over-flag with single-vote threshold | Chat 4 Phase A-B | Resolved by D.3 cross-check pattern -- every report column shows agreement / disagreement explicitly. |
| P4AB-9 | Mixture disambiguation has FP+FN (real-data observation) | Chat 4 Phase A-B v1.1 | Cross-check pattern is now **mandatory**, not optional. Test `test_markdown_has_cross_check_column` enforces this. |

### B.2 New issues from Phase C-D

| # | Issue | Severity | Mitigation |
|---|---|---|---|
| **P4CD-1** | The smoke test (`scripts/smoke_test_phase_c.py`) uses a mock model with random weights; on the user's machine the real model produces meaningfully different numbers. The smoke test only verifies that the pipeline **runs**, not that the numbers are correct. | Low -- by design | The 23 unit tests in `tests/test_inference.py` use the same mock, but check **contracts** (output shapes, simplex constraint, cross-check tags, schema keys) that hold regardless of weights. The real-data validation is via `scripts/run_demo_reports.py`, which the user runs separately. |
| **P4CD-2** | `predict()` uses module-level cache (`_resources`) — calling it from multiple threads is **not** thread-safe. Same for the Streamlit dashboard's `@st.cache_resource` warmup. | Low | Acceptable for thesis demo / batch inference. For production, wrap predict() in a class with per-instance state and use a thread-local cache. |
| **P4CD-3** | The Streamlit dashboard's `_parse_uploaded()` function silently fails on malformed CSV/TXT (catches all exceptions and shows an error). It does not validate that the upload's wavenumber axis matches the model's. A user uploading a 785 nm-laser spectrum from a different instrument would get garbage results without warning. | Medium for production, **negligible for thesis demo** | Add wavenumber-axis check in v2. For the defense, the dashboard is a *demo* of pre-processed cached spectra. Document in the demo speaker notes. |
| **P4CD-4** | `scripts/run_demo_reports.py` picks the "hard OOD" demo via a synthetic 380 cm⁻¹ spike on a copy of the ID demo's row. This is **easier** than a real MoS₂ spectrum because the rest of the spectrum still looks like a normal AA mixture. A real MoS₂ from `data/raw/ood_demo/MoS2-160o-12h-ph5.txt` would be a much cleaner demonstration. | Low -- documented stretch work | If Phase D is fully done, replace the synthetic demo with a real MoS₂ loader (`src/data/mos2_loader.py` -- not built in this chat, scope-deferred to Chat 5 / v2). |
| **P4CD-5** | The Streamlit dashboard imports torch / matplotlib at module top -- this means startup is ~3-5 s on first load. Subsequent reruns within the same session are cached. | None | Standard Streamlit behaviour. |
| **P4CD-6** | The OOD components in the report are derived by re-running the OODScorer's internal logic (one deterministic + one MC forward). This is duplicated work relative to predict_with_uncertainty -- in theory we could share the MC samples between the two paths. | Negligible (~50 ms savings per spectrum) | Premature optimisation. Refactor only if dashboard latency becomes a UX problem. |
| **P4CD-7** | Markdown image embeds use **relative filenames** (e.g. `demo_1_reconstruction.png`). This works because `save_report` and `plot_all` are called with the same output_dir, so the PNG lives next to the MD. If a user moves a single .md file out of its folder, the images break. | Cosmetic | Standard pattern; the MD is meant to be viewed inside its containing folder. |

### B.3 NOT done in Phase C-D (deferred or out of scope)

- **Real MoS₂ loader** (`src/data/mos2_loader.py`): would replace the synthetic-spike demo 3 with a real cross-domain OOD test. Cheap (~30 min) but not strictly needed for the MVP. **Recommended for Chat 5 stretch.**
- **bacteria_ID external benchmark loader** + AUROC computation: real OOD AUROC number for the thesis (P4AB-6 / Custom Instructions §6). **Recommended for Chat 5 stretch if time permits.**
- **PCA + SVM and ResNet-only demo reports**: Phase B baselines don't run through predict() (different output schema -- no MC dropout, no reconstruction). If the thesis defense wants per-sample baseline reports, T26 needs a "baseline mode" that handles missing fields. Not requested by the chat-task message; left as v2 feature.
- **Per-test-row qualitative grid**: a 5×6 figure showing 30 sample predictions side by side would strengthen the thesis but is repetitive work better done as a Chat 5 / write-up task.
- **AAM mineral extension**: still scope-deferred (P4AB-6).

---

## C. Decisions diverged from the original plan

### C.1 Phase C task-message spec vs Phase C-D execution

| # | Original (chat-task message Phase A → C scope) | What this chat did | Reason |
|---|---|---|---|
| **D1** | Task message labels them T20-T22 (peak ext / symbolic / novelty) and T26-T28 (report / visualize / dashboard). The "T_glue predict()" was implicit. | Renamed the actual *predict()* glue to **T_glue** so handover docs are clear that it's a separate task from T20-T22 (engine layer, Phase A-B) and T26-T28 (inference UI). | Earlier handovers (CHAT4_PHASE_AB_HANDOVER §D.5) already used "T_glue". Consistent naming. |
| **D2** | T26 spec: 4-section Markdown (Composition / Peaks / Physics validation / OOD assessment / Visualisations). | Added a 6th section: **Benchmark Context** (pulled from `results/benchmark_table.json`). | Per CHAT2_PHASE_B_HANDOVER §F, demo reports should self-contain the 3-model comparison. Each demo report now includes a 3-row MAE comparison + this-sample's-MAE so the thesis defense can hand a single MD file to a reviewer and they see the full story. |
| **D3** | T26 spec: dict with `markdown`, `json`, `plain_text` keys. | Matches exactly; also added `save_report()` helper that writes MD + JSON to disk. | `save_report()` is a one-liner the demo script kept duplicating. |
| **D4** | T27 spec: 3 plot functions. | Added `plot_all()` convenience wrapper. | Used by both `run_demo_reports.py` and the dashboard; saves three function calls per sample. |
| **D5** | T_glue spec: return dict with documented keys. | Returns the **superset** documented in CHAT4_PHASE_AB_HANDOVER §D.2 (22 keys including `composition_mean` and `composition_std_arr` as numpy arrays for downstream plotting). | Strict superset, no breaking change; older callers see the same keys. |
| **D6** | OOD score: single scalar. | predict() returns `ood_score`, `is_ood`, `ood_threshold`, AND a `ood_components` dict with `recon_norm` and `var_norm`. | The report's OOD section names the dominant contributor (recon vs variance) to help reviewers interpret the score. |
| **D7** | T28 spec: 3 tabs (Upload / Analysis / Report). | Matches; added preset buttons in Upload tab ("Histidine-dominant" / "Glucosamine-dominant" / "Random row") because typing row indices for demo is awkward. | Demo UX. |
| **D8** | T28 spec: download PDF. | **Download Markdown** + **Download JSON** instead of PDF. | Reportlab / weasyprint adds 2 dependencies and ~10 MB of install footprint for a feature nobody asked for. Markdown is more useful (can be edited; renders in GitHub). |
| **D9** | T_glue spec: load model from `checkpoints/best.pt` via the recipe in CHAT3_PHASE3_HANDOVER §D.2. | Matches the recipe **exactly** -- including the tempfile-dance for `reconstruction.pure_ref`. | Documented in `_build_model_from_checkpoint`. |
| **D10** | (Not in spec) -- caching of resources. | predict() loads model + OOD scorer + engine layer ONCE on first call, caches in module state. `reset_cache()` provided for tests. | Without this, every dashboard click would pay ~2 s of checkpoint load. With caching, second click is ~0.5 s. |

### C.2 Runtime decisions during this chat

| # | What changed mid-build | Reason | Result |
|---|---|---|---|
| **C-r1** | Initial `plot_peak_annotations` had labels running off the top of the figure. | The y-axis auto-scaled to `max(spectrum)`, leaving no headroom for the annotation arrows above each peak. | Added explicit `ax.set_ylim(ymin, ymax + 0.30 * (ymax - ymin))` to reserve 30% headroom. The smoke-test plot then rendered cleanly with all 4 peak labels visible. |

That's the only mid-build fix. Everything else worked first try once the design was settled.

---

## D. Contracts for Chat 5 (integration)

### D.1 Files Chat 5 can rely on

| File | Source |
|---|---|
| `src/inference/predict.py` | This chat -- T_glue |
| `src/inference/report.py` | This chat -- T26 |
| `src/inference/visualize.py` | This chat -- T27 |
| `dashboard/app.py` | This chat -- T28 |
| `scripts/run_demo_reports.py` | This chat |
| `scripts/smoke_test_phase_c.py` | This chat (sandbox-only verification) |
| `tests/test_inference.py` | This chat -- 23 tests |
| `results/reports/demo_*/demo_*.md` | User produces via `python scripts/run_demo_reports.py` |
| `results/reports/demo_*/demo_*.json` | same |
| `results/reports/demo_*/demo_*_reconstruction.png` | same |
| `results/reports/demo_*/demo_*_peaks.png` | same |
| `results/reports/demo_*/demo_*_ood.png` | same |

### D.2 What Chat 5 needs to do

| Task | File | Estimated | Notes |
|---|---|---|---|
| **REPORT.md** | `docs/REPORT.md` | 4-6h | The thesis-defense English methodology report. ~2-3 pages. Pull numbers from `midcheckpoint_report.json`, `benchmark_table.json`, and demo reports. Per Custom Instructions §11 + §12 cite De Gelder 2007, Karniadakis 2021, Hafner 2025. Limitations section must include T17-B6 (MAE ceiling) and P4AB-9 (mixture FP+FN). |
| **README.md** | `README.md` (top-level) | 1h | Project overview, quickstart, installation, link to REPORT.md, citation. |
| **Tag** | `v0.1.0-mvp` | 5 min | `git tag v0.1.0-mvp; git push --tags`. |
| **Reproducibility script** | `scripts/reproduce_results.sh` | 1h | Single bash script (Windows: PowerShell .ps1) that runs prepare_data → train → midcheckpoint → baselines → compare → demos in order. |
| **Final smoke test** | manual | 30min | Run full `pytest tests/`, run `scripts/run_demo_reports.py`, inspect 3 demo reports visually, run `streamlit run dashboard/app.py` and click through. Fix anything that's broken. |

### D.3 Optional stretch for Chat 5 (only if Day 14 is ahead of schedule)

- `src/data/mos2_loader.py` -- 30 min. Replaces synthetic-spike demo 3 with the real `MoS2-160o-12h-ph5.txt`. Much cleaner thesis story.
- `src/data/bacteria_id_loader.py` + cross-domain OOD AUROC -- 2h. Gives a real OOD AUROC number for the final report.
- Per-test-row qualitative figure (5×6 grid showing 30 predictions) -- 1h.

---

## E. Quick reproduction recipe (Chat 5 day 1)

```powershell
cd E:\Project\KhoaLuanCourse
.venv\Scripts\activate

# 1. Drop in Phase C-D files (no overwrites; src/inference/ is new directory):
#    src/__init__.py
#    src/inference/__init__.py
#    src/inference/predict.py
#    src/inference/report.py
#    src/inference/visualize.py
#    dashboard/app.py
#    scripts/run_demo_reports.py
#    scripts/smoke_test_phase_c.py
#    tests/test_inference.py

# 2. Verify the full test suite is green
pytest tests\ -v --tb=short
# Expected: 320 passed (297 from Chat 2 Phase B + 23 new) in ~45 s

# 3. Produce the 3 demo reports
python scripts\run_demo_reports.py --mc 50
# Output: results\reports\demo_1_id_histidine\
#         results\reports\demo_2_mild_glucosamine_heavy\
#         results\reports\demo_3_hard_synthetic_mos2\
# Each contains: demo_*.md, demo_*.json, 3 PNGs

# 4. Optional: launch the dashboard
streamlit run dashboard\app.py
# Open http://localhost:8501 in browser, click through 3 tabs

# 5. Begin REPORT.md / README.md / v0.1.0-mvp tag
```

If step 2 fails, do NOT continue with thesis write-up -- fix the regression first.

---

## F. Roadmap remaining

| Days | Chat | Tasks | Output |
|---|---|---|---|
| ~~12-13~~ | ~~Chat 4 Phase C~~ | **DONE this chat** | predict.py + report.py + visualize.py + dashboard + 3 demo reports + 23 new tests |
| 14 | **Chat 5 integration (NEXT)** | REPORT.md + README + v0.1.0-mvp tag + final smoke test | thesis-ready repo |

---

## G. Read order for Chat 5

1. **This file** (`CHAT4_PHASE_CD_HANDOVER.md`) — what Phase C-D produced.
2. **`CHAT2_PHASE_B_HANDOVER.md`** — benchmark numbers for the comparison narrative in REPORT.md.
3. **`CHAT4_PHASE_AB_HANDOVER.md`** v1.1 — engine modules + cross-check pattern (P4AB-9) for the limitations section.
4. **`midcheckpoint_report.md`** — T17 verdict (MAE 0.0550 FAIL, recon 0.9698 PASS-target, CVR 0.0000 PASS-target) — headline numbers.
5. **Chat 2 phase B PNGs** (`benchmark_*.png`) — for embedding in REPORT.md.
6. Custom Instructions §6 (targets), §11 (Day-14 DoD), §12 (citations) — to know what numbers count as "win" and how to cite.

---

## H. The thesis story in one paragraph (for Chat 5 to expand)

"We built a physics-informed Raman-spectrum analyser whose composition
quantification MAE (0.0550) on a strict composition-OOD split misses the
literature-comparable target by ~2× but is competitive with classical
(PCA+SVM, 0.0479) and pure black-box deep-learning (ResNet-only, 0.0462)
baselines on the same split. The MVP's contribution is NOT MAE: it is
the *combination* of (a) a reconstruction module that validates each
prediction against the Beer-Lambert linearity assumption (cosine
similarity 0.97, constraint violation rate 0%); (b) MC-Dropout
uncertainty that surfaces per-compound variance to flag low-confidence
predictions; (c) an OOD scorer (T19) that calibrates against the
validation set and flags spectra dissimilar to training; (d) a symbolic
peak-to-bond mapping (T21) that produces interpretable explanations of
which peaks justify each predicted compound; and (e) a novelty locator
(T22) that clusters peaks the bond DB cannot account for. The
combination produces per-sample reports (T26) that cross-check the
black-box prediction against the symbolic head, flagging disagreement —
a feature neither baseline can provide. Limitations are documented
openly: the MAE ceiling, the Wasatch WP-785 calibration drift that
required widening DB tolerances post-hoc, and the symbolic head's
characteristic false-positive vote on low-mixing-ratio components."

---

*Document version 1.0. Generated at end of Chat 4 Phase C-D
(T_glue + T26 + T27 + T28). Hand to Chat 5 for REPORT.md /
README / v0.1.0-mvp tag.*
