# SOTA Review — 10 Baseline Models from Prior Work (HNKHSV)

> **Purpose.** Document the state of the prior undergraduate research project
> (10 models, 2 split protocols), identify *why* every model fails on
> compositional out-of-distribution evaluation, and articulate the precise
> gap that this MVP fills.
>
> **Source.** "Ứng dụng Học Máy trong Phân tích Phổ Raman — Nghiên cứu So
> sánh các Mô hình cho Hỗn hợp Axit Amin" (HNKHSV 2024-2025). Stage 1 & 2
> deliverables: 10 model implementations + benchmark.
>
> **Audience.** Anyone reading the thesis defense or the methods paper. Reads
> in 5-10 minutes.

---

## 1. Prior work in one paragraph

The HNKHSV project trained ten regression models to predict the molar
fractions of six compounds (Ala, Asn, Asp, Glu, His, GlcN) from a single
1024-channel Raman spectrum. Each model was evaluated under two protocols:
sample-level random 60/20/20 and vial-level "group" 42/6/6. The benchmark
was honest: the random protocol yielded **R² = 0.79–0.91** on test, while
the same models on the group protocol collapsed to **R² = 0.22–0.64**
— a documented generalisation gap of roughly 30 percentage points. The
authors correctly framed this as evidence that "high R² ≠ understanding
the spectrum" and identified the need for additional evaluation metrics
beyond regression accuracy. That observation is the entry point for the
present MVP.

## 2. The ten models, organised by paradigm

| # | Name | Paradigm | Key idea |
|---|---|---|---|
| M01 | Ridge + Softmax | Linear chemometrics | Ridge regression on log-targets, softmax read-out, α swept over 11 values |
| M02 | NMF + PLSR | Classical unmixing | NMF(K=8) concentrations + bond integrals → standardise → PLSR(LV swept) |
| M03 | NNLS + GBR | Two-stage hybrid | NNLS unmixing against pure-component references → GBR corrects residual |
| M04 | MLP | Deep baseline | 1024 → 256 → 128 → 64 → 6, KL-div loss, AdamW, mixup + Gaussian noise |
| M05 | 1D ResNet | Deep baseline | Strided 1D-CNN with two residual blocks, KL-div loss, noise augmentation |
| M06 | Adaptive Preprocessing | Multi-task | Gate network selects per-spectrum among ['none', 'snv', 'sg_snv'], reconstruction + total-variation losses |
| M07 | Bond-attention | Multi-task | Spectral encoder + bond-feature encoder + multi-head attention; auxiliary peak-position / peak-intensity heads |
| M08 | Multi-task chemistry | Multi-task | Encoder + learnable "pure components" + chemistry head (pH, polarity, hydrophilicity, bond strength) |
| M09 | HP search + pruning | Meta-learning | Random search over (loss_fn, lr, wd, dropout, hidden) for 6 trials × 15 epochs, then 20% magnitude pruning |
| M10 | RIER | Hybrid radial | 8 parallel "spokes" (VAE, PCA, FFT, BondStats, Derivatives, NMF, Moments, XCorr) + gated fusion |

Three families:
- **M01-M03** classical chemometrics (interpretable in the linear-algebra
  sense, but no physics)
- **M04-M05** deep regression (black box with regularisation tricks)
- **M06-M10** "explainability bolt-ons" — auxiliary losses, attention
  weights, gating networks, post-hoc bond-detection plots

## 3. Black-box weakness analysis — model by model

The review below applies one consistent rubric:

> A model is **black-box** if its predictions cannot be locally
> attributed to (i) specific spectral peaks, (ii) named chemical bonds,
> and (iii) a physics-justified composition law that an analyst can
> falsify. Auxiliary losses, attention maps, and post-hoc plots do not
> count — they describe the model's *behaviour*, not its *commitments*.

### M01 (Ridge + Softmax)
A 6 144-parameter linear regressor over the full spectrum. Coefficients are
inspectable but *non-localised*: each amino acid has a 1024-long weight
vector with no constraint that high-magnitude weights coincide with
chemically meaningful peaks. Softmax read-out is applied post-hoc, so the
simplex constraint Σα_i = 1 is enforced *after* fitting, not during.
*Weakness:* No physics commitment; "interpretability" reduces to plotting
weight curves with no causal claim.

### M02 (NMF + PLSR)
NMF decomposes the data matrix into K=8 non-negative basis spectra and
concentrations. The basis vectors are not constrained to match the six
known compounds, and K=8 ≠ 6 — the "extra" components absorb baseline
residue and noise. PLSR then mixes NMF concentrations with bond integrals
through latent variables that have no chemical interpretation.
*Weakness:* The "pure" basis is learned, not measured. Predictions cannot
be traced back to physical compounds even though the linear-algebra is
transparent.

### M03 (NNLS + GBR)
The strongest classical baseline. Stage 1 uses *measured* pure spectra as
the dictionary — this is genuine domain knowledge and is why M03 leads on
the group split. But stage 2 is a 100-tree GBR over PCA features that
"corrects" stage-1 residuals; the correction is opaque, and there is no
mechanism preventing it from undoing the physics-respecting first stage.
*Weakness:* The interpretable part (NNLS) and the predictive part (GBR)
are decoupled; the second stage destroys the first stage's auditability.

### M04 (MLP)
A 1024 → 256 → 128 → 64 → 6 MLP with batch-norm, dropout, mixup, and
Gaussian noise. Trained with KL divergence on the simplex output. There
is no peak-level intermediate representation, no physics constraint, no
mechanism for refusal. Top performer on the random split (R² = 0.91)
because it can memorise per-spectrum quirks; on the group split it loses
30 R² points because nothing forces it to use peaks rather than baseline
shape.
*Weakness:* Pure black-box. No interpretability claim was ever made.

### M05 (1D ResNet)
Same loss and training schedule as M04 but with 1D convolutions and skip
connections. Convolutional kernels are sometimes presented as
"interpretable" via Grad-CAM or saliency, but no such analysis was
performed in the prior work. Performance is comparable to M04 on random
split, slightly weaker on group.
*Weakness:* Identical to M04 — convolution is a form of weight sharing,
not interpretation.

### M06 (Adaptive Preprocessing)
A gate network selects a soft combination of preprocessing variants
('none', 'snv', 'sg_snv'). Adds a reconstruction loss and a total-variation
penalty. Argued to "understand" the spectrum because it learns *which*
preprocessing helps. But the per-spectrum gate is not constrained to be
physically meaningful, and the downstream MLP head still treats the
fused spectrum as a featureless 1024-D vector.
*Weakness:* "Adaptive" is a function-approximation argument, not a
chemistry argument. The model learns *that* SNV helps; it does not
articulate *why*.

### M07 (Bond-attention)
A real attempt at interpretability: the model receives the raw spectrum
*and* a bond-feature vector (5 numbers per bond region × N regions),
combines them through multi-head attention, and is supervised by
auxiliary peak-position and peak-intensity targets. The attention weights
are plotted post-hoc.
*Weakness:* The attention is over fixed bond regions, but the regions
themselves were defined manually and the network has no incentive to
align attention with chemistry — only with prediction accuracy. The
auxiliary losses use peak features extracted by the same `find_peaks`
routine the network is meant to learn from, making the supervision
near-circular.

### M08 (Multi-task chemistry)
Encoder + learnable "pure components" matrix + a chemistry head trained
to predict pH, polarity, hydrophilicity, bond strength. The chemistry
labels are themselves derived from the same input spectrum by a
deterministic formula in `extract_chemistry()`, so the auxiliary loss
mostly forces the encoder to memorise that formula — it adds no external
information. The "pure components" are randomly initialised and trained
freely; nothing pins them to the six measured pure spectra.
*Weakness:* Multi-task learning without external grounding becomes
multi-loss self-distillation. The chemistry head looks principled but
contributes no new constraint.

### M09 (HP search + pruning)
A meta-procedure. 6 trials × 15 epochs of random search over (loss
function, learning rate, weight decay, dropout, two hidden sizes), then
the winning configuration is retrained for 40 epochs and pruned to 80%
density. Improves robustness slightly on random split. Does *nothing* for
interpretability — the winning model is still an MLP.
*Weakness:* Meta-learning addresses optimisation, not understanding.

### M10 (RIER)
The most ambitious. Eight parallel "spokes" extract complementary
representations (VAE latent, PCA-like projection, FFT magnitudes, bond
features, derivative features, NMF concentrations, statistical moments,
cross-correlation), a gate network reweights them, and the fused
representation feeds a softmax head. The flower-shaped DAG visualisation
is a real engineering artefact. But the gate network learns to *suppress*
several spokes (VAE, BondStats, NMF were dropped per slide 18) — exactly
the spokes that carry chemical content.
*Weakness:* The gate optimises validation loss, not chemical fidelity.
A model that "understands the spectrum" should *increase* the weight of
bond features when bonds are diagnostic; M10 learned to ignore them
because PCA + moments give better R². This is the clearest in-prior-work
demonstration that auxiliary information needs hard structural
constraints, not soft gates.

## 4. The benchmark, in numbers

Reported R² (slide 14, 15) on the test set, sorted by group-split rank:

| Model | Random R² | Group R² | Generalisation gap |
|---|---|---|---|
| M03 (NNLS + GBR) | 0.903 | **0.640** | 0.263 |
| M08 | 0.903 | 0.628 | 0.275 |
| M07 | 0.905 | 0.622 | 0.283 |
| M04 (MLP) | **0.910** | 0.610 | 0.300 |
| M06 | — | 0.592 | — |
| M10 (RIER) | 0.900 | 0.580 | 0.320 |
| M01–M02, M05, M09 | 0.79–0.89 | 0.22–0.55 | 0.27–0.50 |

The single most informative number is M03's group-split lead: it is the
only model whose interpretable component (NNLS against measured pure
spectra) is *first* in the pipeline, and it is the only model whose group
performance approaches its random performance. Every model that puts
interpretability *after* a free-form predictor (M07's attention, M08's
chemistry head, M10's gate) loses to M03 on the OOD protocol despite
adding parameters and auxiliary losses.

## 5. The four gaps this MVP fills

The prior work's authors named the missing pieces themselves on slide 4
("Thiếu: đánh giá tổng quát hóa, tích hợp kiến thức hóa lý, so sánh
phương pháp, metric 'hiểu phổ'"). The MVP turns each of those four
items into an engineering commitment.

### Gap 1 — Physics-respecting reconstruction
*Prior work:* every model fits compositions directly; the predicted
α-vector is never required to *explain* the input spectrum. M02's NMF and
M08's "pure components" learn their own basis, M03's NNLS uses real pure
spectra but is then overruled by GBR.
*MVP commitment:* the prediction `α` must satisfy
`s ≈ Σ α_i · scale_i · pure_i` where `pure_i` are the *measured* mean
pure spectra (from `engine/reference_spectra.npy`, T03+). Reconstruction
error enters the loss as `β·MSE(s_recon, s_input) + λ·(1−cos(s_recon, s_input))`
and at inference time becomes the OOD score's first ingredient. The
predicted simplex is *physically falsifiable* — if it doesn't reconstruct
the spectrum, it's wrong by definition.

### Gap 2 — Out-of-distribution detection, not just OOD evaluation
*Prior work:* split A' (random) vs split A (composition) demonstrated
the gap but offered no per-sample mechanism for flagging unfamiliar
spectra. Every model returns a confident simplex regardless of whether
the input came from a vial it has seen before.
*MVP commitment:* MC Dropout (50 forward passes) provides predictive
variance; combined with reconstruction error, the OOD score
`s_ood = w_r·E_recon + w_v·Var_pred` is calibrated on the validation set
(95th-percentile threshold) and a binary `novelty_flag` is returned with
every prediction. AUROC on the held-out composition test set is the
headline OOD metric, alongside ID Acc/MAE.

### Gap 3 — Symbolic peak → bond mapping (the interpretability anchor)
*Prior work:* M07 plots attention weights, M08 trains chemistry heads,
M10 visualises spoke gates — all *post-hoc*, all learned. None of them
can answer "why did you predict 30 % Histidine?" with a sentence the
analyst can verify against Socrates' tables.
*MVP commitment:* an *editable JSON database* (`engine/bond_mapping.json`,
30 seed entries from Socrates 3rd ed. + De Gelder 2007) maps each peak
wavenumber to a chemical bond and a list of compounds. The
`BondMapper.match_peak()` lookup is deterministic, takes ~1 µs, and is
the same operation an analyst would perform with a printed handbook.
Every detected peak in the inference report carries a bond name; peaks
not in the DB raise a novelty flag for human review. **This is the
project's core differentiator.**

### Gap 4 — A "spectrum-understanding" metric
*Prior work:* slide 18 explicitly proposes "Peak Alignment Score" as a
future metric; never implemented.
*MVP commitment:* the Constraint Violation Rate (CVR) and the
reconstruction cosine similarity become first-class metrics in
`src/eval/metrics.py`, reported alongside MAE and AUROC. CVR ≤ 5 % and
median cos ≥ 0.95 are explicit success criteria.

## 6. What we deliberately keep from prior work

Three pieces of prior work transfer wholesale (not reinvented):

1. **The classical preprocessing pipeline** — Asymmetric Least Squares
   baseline, cosmic-ray clipping, Savitzky-Golay smoothing, SNV
   normalisation. Ported in `src/data/preprocess.py` from the prior
   `utils.py` with the same hyperparameter ranges (lam = 1e5, p = 0.01,
   SG window 11/3, threshold 5σ). Validated on real data:
   per-row mean ≈ 0, std = 1.000 ± 1e-3 across all 4378 rows.

2. **The 1D-ResNet backbone (M05)** — kept as the feature extractor.
   The architectural lesson from M04 vs M05 was that strided 1D
   convolutions reach competitive accuracy with fewer parameters than
   an MLP and translate-equivariance helps with the random ±10 cm⁻¹
   peak-shift augmentation. Adopted into `src/models/backbone.py` (T11).

3. **The `vial #` group-split protocol** — same GroupShuffleSplit-style
   logic, generalised to support three schemes (A composition / A'
   random / B component) in `src/data/splits.py` and persisted to JSON
   for reproducibility. Note that scheme B turned out to be impractical
   on this dataset (every compound appears in 49/54 vials, leaving a
   50-row train pool); documented as "stress test" in `data_config.yaml`.

## 7. What we deliberately do *not* reproduce

* **No NMF basis-learning, no learnable "pure components".** The
  reconstruction module uses the measured pure spectra only.
* **No auxiliary chemistry-property regression.** Self-supervised targets
  derived from the input spectrum cannot add information the input did
  not already contain (M08's lesson).
* **No multi-spoke gated fusion.** A gate that can suppress chemical
  content in favour of statistical features is not interpretable, and
  M10's empirical result (BondStats spoke disabled) confirms the
  failure mode.
* **No attention-as-explanation.** Attention weights are not
  interpretation; they are differentiable indices that happen to be
  visualisable.
* **No hyperparameter search-and-prune.** Meta-optimisation does not
  address the interpretability gap and consumes the sprint budget.

## 8. References (cite from REPORT.md)

- HNKHSV 2024-2025 Stage 1 & 2 deliverable (this project's prior
  state). 22 slides, 10-model benchmark on 4378 spectra.
- Zarei *et al.* (2023). *Anal. Chem.* 95(43): 15908–15916. The single
  external reference cited in the prior work.
- De Gelder *et al.* (2007). *J. Raman Spectrosc.* 38(9): 1133–1147.
  Reference Raman spectra of biological molecules; basis of the bond
  database tolerances.
- Socrates G. (2004). *Infrared and Raman Characteristic Group
  Frequencies*, 3rd ed. Wiley. Source for the 30 seed entries in
  `engine/bond_mapping.json`.
- Karniadakis *et al.* (2021). *Nat. Rev. Phys.* 3, 422–440.
  Methodological framing for physics-informed machine learning.

---

*Author: T01 sprint, Day 1. Reviewed against `model01.py` … `model10.py`,
`run_all_models.py`, `run_random_split.py`, and HNKHSV.pptx slides 4–18.*
