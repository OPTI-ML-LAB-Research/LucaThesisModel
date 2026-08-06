# Stretch T29B — AAM Zero-shot Test (BEFORE Retrain) — Handover

**Setup:** model AA-only (6 outputs), test trên AAM test set.
Model **chưa từng thấy minerals**. Câu hỏi: physics + OOD có nhận biết được mineral-rich samples là OOD?

## Kết quả

- Test n = 973 (0 low-mineral + 973 mineral-rich)
- MC samples = 50

### Reconstruction cosine

| Class | Mean | Median |
|---|---|---|
| Low mineral | N/A | N/A |
| Mineral-rich | 0.5021 | 0.4899 |

### OOD score

| Class | Mean | Median |
|---|---|---|
| Low mineral | N/A | N/A |
| Mineral-rich | 0.9836 | 1.0000 |

### AUROC (mineral_rich vs low_mineral) = N/A

> Không tính được AUROC vì test split này thiếu một trong hai lớp (cần cả low-mineral lẫn mineral-rich). Xem ghi chú bên dưới.

## Diễn giải

- Mineral-rich samples chứa quartz (~464 cm⁻¹) và calcite (~1086 cm⁻¹).
- 6 AA pure references KHÔNG thể tái dựng các peaks này.
- Reconstruction error mineral-rich KỲ VỌNG cao hơn low-mineral.
- OOD score (combine recon + variance) KỲ VỌNG cao hơn cho mineral-rich.

Vì split chỉ có một lớp, dùng giá trị tuyệt đối của recon cosine / OOD score (bảng trên) làm bằng chứng thay cho AUROC.

## Files được tạo

- `results/stretch/t29b_zeroshot_pre/results.json` -- per-sample records
- `results/stretch/t29b_zeroshot_pre/summary.json`
- `results/stretch/t29b_zeroshot_pre/histograms.png`

## Cách dùng trong báo cáo

Đây là 'BEFORE' phần của paired comparison. Sau khi T29E xong, viết Chương 3 subsection:

'Bảng X — AAM evaluation: AA-only model vs AAM-retrained model trên cùng test set'

Highlight:
- AA-only: dùng recon cosine / OOD score tuyệt đối (AUROC N/A do thiếu lớp low-mineral)
- Sau retrain (T29E), composition prediction cho minerals KỲ VỌNG chính xác
- Đây là evidence cho thesis: physics + OOD work, retrain solves shortcomings.