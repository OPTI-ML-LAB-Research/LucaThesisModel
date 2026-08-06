# Raman Spectrum Analysis Report

- **Sample ID:** `Beryl (Beryl_01 (12).txt)`
- **Date:** 2026-05-30 16:47:58 UTC
- **Model Version:** v0.1.0-mvp
- **n MC samples:** 30

## 1. Composition Analysis

Composition predicted by the learned head (mean ± MC std), with the symbolic head's cross-check tag (per CHAT4_PHASE_AB_HANDOVER §D.3).

| Compound | Predicted (mean) | Uncertainty (±std) | Symbolic vote | Cross-check |
|---|---|---|---|---|
| Alanine | 0.047 (4.7%) | ±0.011 | — | ✓ absent |
| Asparagine | 0.244 (24.4%) | ±0.018 | — | ⚠️ learned-only |
| Aspartic Acid | 0.224 (22.4%) | ±0.015 | — | ⚠️ learned-only |
| Glutamic Acid | 0.224 (22.4%) | ±0.013 | — | ⚠️ learned-only |
| Histidine | 0.102 (10.2%) | ±0.017 | — | ⚠️ learned-only |
| Glucosamine | 0.159 (15.9%) | ±0.011 | — | ⚠️ learned-only |

**Symbolic head ⇒ no compounds reached the vote threshold.**  (See peak table for individual matches.)

**Uncertainty summary:** predictive entropy = 1.6837, mean compound std = 0.0141.

## 2. Peak Analysis

Detected **5 peaks**. **0** matched to known compounds, **5** unmatched.

| Position (cm⁻¹) | Intensity | FWHM | Bond / Mode | Compound | DB-id | Conf. | Δν |
|---|---|---|---|---|---|---|---|
| 323.8 | 0.959 | 15.0 | — | — | — | none | — |
| 398.7 | 1.360 | 17.6 | — | — | — | none | — |
| 687.7 | 5.924 | 19.6 | — | — | — | none | — |
| 1017.0 | 0.787 | 21.3 | — | — | — | none | — |
| 1071.1 | 7.981 | 24.7 | — | — | — | none | — |

## 3. Physics Validation

- **Reconstruction cosine similarity:** 0.0701  (✗ Floor missed)
- The predicted composition cannot reproduce the input spectrum (cosine sim below floor 0.85). **Constraint violated** -- this prediction should be treated with extreme caution.

## 4. OOD Assessment

- **OOD Score:** 0.8912  (threshold 0.9202)
- **Verdict:** ✓ IN-DISTRIBUTION
- **Components:** recon_norm = 1.000, var_norm = 0.728
- **Novel peaks:** 5 unmatched, grouped into 5 cluster(s).

**Cluster details:**
- cluster centroid 324 cm-1 (1 peak, span 0 cm-1): lattice / skeletal modes, metal-X / heavy-atom stretches  --  inorganic phonons (MoS2 E2g ~380, A1g ~408); metal-O stretches; framework breathing; M-S, M-O, M-Cl, M-N inorganic stretches; skeletal deformation
- cluster centroid 399 cm-1 (1 peak, span 0 cm-1): lattice / skeletal modes, metal-X / heavy-atom stretches  --  inorganic phonons (MoS2 E2g ~380, A1g ~408); metal-O stretches; framework breathing; M-S, M-O, M-Cl, M-N inorganic stretches; skeletal deformation
- cluster centroid 688 cm-1 (1 peak, span 0 cm-1): C-S, C-X bend; ring deformation  --  thiols, halogenated alkanes, aromatic ring deformation
- cluster centroid 1017 cm-1 (1 peak, span 0 cm-1): C-C / C-O stretch  --  alkane C-C; alcohols / ethers C-O; sugar pyranose ring; P-O
- cluster centroid 1071 cm-1 (1 peak, span 0 cm-1): C-C / C-O stretch  --  alkane C-C; alcohols / ethers C-O; sugar pyranose ring; P-O
