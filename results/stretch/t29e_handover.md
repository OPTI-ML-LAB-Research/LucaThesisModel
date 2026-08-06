# Stretch T29E — AAM Post-retrain Test — Handover

**Setup:** model retrain trên AAM-train (T29D, 7 outputs = 6 AA + Minerals), test trên cùng AAM-test set như T29B.

## Kết quả tổng quan

- n test = 973
- Overall MAE = **0.0126**
- Recon cosine mean = **0.6048**

### MAE per-compound

| Compound | MAE |
|---|---|
| Alanine | 0.0056 |
| Asparagine | 0.0115 |
| Aspartic Acid | 0.0179 |
| Glutamic Acid | 0.0062 |
| Histidine | 0.0114 |
| Glucosamine | 0.0056 |
| Minerals | 0.0299 |

### Reconstruction cosine theo class

- All samples: mean = 0.6048
- Low mineral: mean = N/A
- Mineral-rich: mean = 0.6048

### Minerals MAE breakdown

- Trong samples low-mineral (truth ≈ 0): MAE = N/A
- Trong samples mineral-rich (truth > 0.05): MAE = 0.0299

## Files

- `results/stretch/t29e_posttrain/summary.json`
- `results/stretch/t29e_posttrain/raw.npz`
- `results/stretch/t29e_posttrain/recon_cosine_histogram.png`
- `results/stretch/t29e_posttrain/composition_scatter.png`
- `results/stretch/t29_paired_comparison.md` -- **MAIN paired comparison**

## Cách dùng trong báo cáo

Thêm vào Chương 3 subsection mới:

### 3.X — AAM Evaluation (Paired Comparison)

1. Mô tả dataset AAM (12,956 phổ, 6 AA + 2 minerals)
2. Mô tả setup paired comparison (cùng test set, before vs after retrain)
3. Chèn paired comparison table từ `t29_paired_comparison.md`
4. Chèn 2 hình:
   - `recon_cosine_histogram.png` (after retrain)
   - `composition_scatter.png` (post-train predictions)
5. Diễn giải:
   - Trước retrain: mineral-rich = OOD (recon cosine thấp từ T29B)
   - Sau retrain: minerals học được (MAE từ T29E)
   - Conclusion: modular design + physics + OOD = solid framework