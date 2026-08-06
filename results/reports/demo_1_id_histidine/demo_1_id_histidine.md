# Raman Spectrum Analysis Report

- **Sample ID:** `demo_1_id_histidine`
- **Date:** 2026-05-14 15:52:37 UTC
- **Model Version:** v0.1.0-mvp
- **n MC samples:** 50

## 1. Composition Analysis

Composition predicted by the learned head (mean ± MC std), with the symbolic head's cross-check tag (per CHAT4_PHASE_AB_HANDOVER §D.3).

| Compound | Predicted (mean) | Uncertainty (±std) | Ground truth | Error | Symbolic vote | Cross-check |
|---|---|---|---|---|---|---|
| Alanine | 0.318 (31.8%) | ±0.018 | 0.381 | -0.063 | 0.5 | ⚠️ learned-only |
| Asparagine | 0.061 (6.1%) | ±0.013 | 0.100 | -0.039 | 0.5 | ⚠️ learned-only |
| Aspartic Acid | 0.064 (6.4%) | ±0.012 | 0.022 | +0.042 | — | ⚠️ learned-only |
| Glutamic Acid | 0.020 (2.0%) | ±0.007 | 0.172 | -0.152 | **1.0** | ⚠️ symbolic-only |
| Histidine | 0.394 (39.4%) | ±0.021 | 0.271 | +0.122 | — | ⚠️ learned-only |
| Glucosamine | 0.144 (14.4%) | ±0.014 | 0.054 | +0.090 | 0.5 | ⚠️ learned-only |

**Symbolic head ⇒ likely present:** Glutamic Acid.

**Uncertainty summary:** predictive entropy = 1.4343, mean compound std = 0.0140.

## 2. Peak Analysis

Detected **13 peaks**. **8** matched to known compounds, **5** unmatched.

| Position (cm⁻¹) | Intensity | FWHM | Bond / Mode | Compound | DB-id | Conf. | Δν |
|---|---|---|---|---|---|---|---|
| 406.9 | 1.129 | 16.9 | — | — | — | none | — |
| 542.5 | 1.677 | 14.5 | COO- bend / C-N-C bend | Alanine, Asparagine, Aspartic Acid, Glutamic Acid | P018 | high | +2.5 |
| 651.4 | 0.807 | 10.8 | — | — | — | none | — |
| 853.9 | 6.043 | 15.6 | C-COO- symmetric stretch / C-CH3 stretch | Alanine | P001 | medium | +3.9 |
| 922.1 | 2.408 | 16.8 | C-C-N stretch | Glutamic Acid | P020 | high | -2.9 |
| 1086.7 | 1.817 | 10.1 | C-O stretch / pyranose ring | Glucosamine | P005 | medium | +6.7 |
| 1113.7 | 2.530 | 12.9 | — | — | — | none | — |
| 1140.4 | 1.169 | 13.7 | — | — | — | none | — |
| 1269.3 | 3.320 | 10.1 | Amide III (N-H bend + C-N stretch) | Asparagine | P009 | medium | +4.3 |
| 1314.9 | 5.482 | 14.0 | CH deformation | Alanine, Asparagine, Aspartic Acid, Glutamic Acid | P025 | medium | -5.1 |
| 1352.7 | 3.675 | 19.5 | — | — | — | none | — |
| 1410.4 | 2.914 | 12.7 | COO- symmetric stretch | Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine | P011 | high | +0.4 |
| 1460.8 | 2.627 | 9.8 | NH3+ asymmetric deformation | Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine | P029 | medium | +5.8 |

## 3. Physics Validation

- **Reconstruction cosine similarity:** 0.9654  (✓ PASS-target (≥ 0.95))
- The predicted composition reproduces the input spectrum well via the linear Beer-Lambert combination of pure references. Constraint satisfied.

## 4. OOD Assessment

- **OOD Score:** 0.9368  (threshold 0.9202)
- **Verdict:** ⚠️ **OUT-OF-DISTRIBUTION**
- **Components:** recon_norm = 1.000, var_norm = 0.842
- **Novel peaks:** 5 unmatched, grouped into 4 cluster(s).

**Cluster details:**
- cluster centroid 407 cm-1 (1 peak, span 0 cm-1): metal-X / heavy-atom stretches  --  M-S, M-O, M-Cl, M-N inorganic stretches; skeletal deformation
- cluster centroid 651 cm-1 (1 peak, span 0 cm-1): C-S, C-X bend; ring deformation  --  thiols, halogenated alkanes, aromatic ring deformation
- cluster centroid 1122 cm-1 (2 peaks, span 27 cm-1): C-C, C-N, C-O stretches  --  alkyl C-N (amines); ester / ether C-O; aromatic C-H in-plane bend; sulfate S=O
- cluster centroid 1353 cm-1 (1 peak, span 0 cm-1): CH bending / COO- symmetric stretch  --  aliphatic CH2/CH3 deformation; carboxylate sym stretch; amide III in proteins

## 5. Visualisations

![Reconstruction overlay](demo_1_id_histidine_reconstruction.png)

![Peak annotations](demo_1_id_histidine_peaks.png)

![OOD summary](demo_1_id_histidine_ood.png)

## 6. Benchmark Context

On the same test split (Scheme-A composition-OOD, 540 rows):

| Model | Quant MAE | Notes |
|---|---|---|
| _(no benchmark numbers available yet)_ | -- | run T23/T24/T25 to populate `results/benchmark_table.json` |

**This sample's MAE: 0.0847**.
