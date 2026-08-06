# Stretch T30A — Real MoS₂ Demo — Handover

**Context:** thay synthetic 380 cm⁻¹ Gaussian spike trong demo 3 (Phase C)
bằng phổ MoS₂ thực được tự tay đo. Đây là OOD case 'sạch' nhất vì MoS₂
là chất rắn vô cơ hoàn toàn ngoài domain amino acid.

## Kết quả

- Input: `data/raw/ood_demo/MoS2-160o-12h-ph5.txt`
- MC samples: 50
- Output folder: `results/reports/demo_3_real_mos2/`

### Composition do model dự đoán

Model bị buộc gán MoS₂ vào 6-simplex AA (không có ground truth).
Composition KHÔNG có ý nghĩa hoá học — chỉ là 'chữ ký AA gần nhất'.

| Compound | Predicted | ±Std |
|---|---|---|
| Alanine | 0.112 | ±0.011 |
| Asparagine | 0.168 | ±0.012 |
| Aspartic Acid | 0.196 | ±0.012 |
| Glutamic Acid | 0.287 | ±0.015 |
| Histidine | 0.087 | ±0.012 |
| Glucosamine | 0.150 | ±0.010 |


### Peak detection: 6 peaks detected

- 0 matched với DB
- 6 unmatched (likely lattice modes)

MoS₂ chính peaks (literature):
- E2g mode ~ 380 cm⁻¹
- A1g mode ~ 408 cm⁻¹

---

## Cách dùng kết quả này trong báo cáo

**Trong Chương 3.6 — Demo case studies**: thay 'Demo 3 — Hard OOD Synthetic',
đổi tên thành 'Demo 3 — Real OOD MoS₂', cập nhật:

- Input: phổ MoS₂ tự đo (thay synthetic spike)
- Composition prediction: như bảng trên
- Reconstruction cosine: số mới (kỳ vọng thấp hơn 0.943)
- OOD verdict: nếu bây giờ đúng → success story, nếu vẫn fail → consistent limitation

**Trong Chương 3.7.2 — Hạn chế**: cập nhật mục 3 (OOD detection):
- TRƯỚC: 'OOD detector chỉ test trên synthetic spike, demo 3 false negative'
- SAU: 'OOD detector đã test trên real MoS₂. Verdict: {kết quả thực tế}'

**Hình mới cần chèn vào báo cáo**:
- `results/reports/demo_3_real_mos2/demo_3_real_mos2_reconstruction.png`
- `results/reports/demo_3_real_mos2/demo_3_real_mos2_peaks.png`
- `results/reports/demo_3_real_mos2/demo_3_real_mos2_ood.png`