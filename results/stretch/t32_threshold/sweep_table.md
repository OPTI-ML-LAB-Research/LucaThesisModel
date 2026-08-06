# T32 — Quét trọng số × thang × threshold-F1 (AA=ID vs AAM=OOD)

ID(AA)=973, OOD(AAM)=973. Threshold tối ưu F1 trên chính tập này.

| Thang | recon_w | var_w | AUROC | F1 | Threshold | Recall | FPR |
|---|---|---|---|---|---|---|---|
| tanh (bão hòa mềm) | 0.6 | 0.4 | 0.9947 | 0.9720 | 0.5283 | 0.962 | 0.017 |
| tanh (bão hòa mềm) | 0.7 | 0.3 | 0.9947 | 0.9720 | 0.6164 | 0.962 | 0.017 |
| tanh (bão hòa mềm) | 0.8 | 0.2 | 0.9947 | 0.9720 | 0.7044 | 0.962 | 0.017 |
| tanh (bão hòa mềm) | 1.0 | 0.0 | 0.9947 | 0.9720 | 0.8805 | 0.962 | 0.017 |
| no-clip (giữ độ phân giải far-OOD) | 0.6 | 0.4 | 0.9947 | 0.9718 | 0.9133 | 0.957 | 0.012 |
| no-clip (giữ độ phân giải far-OOD) | 0.7 | 0.3 | 0.9947 | 0.9718 | 1.0655 | 0.957 | 0.012 |
| no-clip (giữ độ phân giải far-OOD) | 0.8 | 0.2 | 0.9947 | 0.9718 | 1.2177 | 0.957 | 0.012 |
| no-clip (giữ độ phân giải far-OOD) | 1.0 | 0.0 | 0.9947 | 0.9718 | 1.5222 | 0.957 | 0.012 |
| clip (gốc repo) | 0.6 | 0.4 | 0.9722 | 0.9641 | 0.5945 | 0.979 | 0.052 |
| clip (gốc repo) | 0.7 | 0.3 | 0.9722 | 0.9641 | 0.6936 | 0.979 | 0.052 |
| clip (gốc repo) | 0.8 | 0.2 | 0.9722 | 0.9641 | 0.7926 | 0.979 | 0.052 |
| clip (gốc repo) | 1.0 | 0.0 | 0.9722 | 0.9641 | 0.9908 | 0.979 | 0.052 |

## Hard-OOD verdict (với threshold F1-optimal mỗi cấu hình)

| Thang | recon_w/var_w | MoS2-160o-12h-ph5 | skmel28_10 |
|---|---|---|---|
| tanh (bão hòa mềm) | 0.6/0.4 | OOD (0.600) | OOD (0.600) |
| tanh (bão hòa mềm) | 0.7/0.3 | OOD (0.700) | OOD (0.700) |
| tanh (bão hòa mềm) | 0.8/0.2 | OOD (0.800) | OOD (0.800) |
| tanh (bão hòa mềm) | 1.0/0.0 | OOD (1.000) | OOD (1.000) |
| no-clip (giữ độ phân giải far-OOD) | 0.6/0.4 | OOD (8.756) | OOD (7.830) |
| no-clip (giữ độ phân giải far-OOD) | 0.7/0.3 | OOD (10.216) | OOD (9.134) |
| no-clip (giữ độ phân giải far-OOD) | 0.8/0.2 | OOD (11.675) | OOD (10.439) |
| no-clip (giữ độ phân giải far-OOD) | 1.0/0.0 | OOD (14.594) | OOD (13.049) |
| clip (gốc repo) | 0.6/0.4 | OOD (0.600) | OOD (0.600) |
| clip (gốc repo) | 0.7/0.3 | OOD (0.700) | OOD (0.700) |
| clip (gốc repo) | 0.8/0.2 | OOD (0.800) | OOD (0.800) |
| clip (gốc repo) | 1.0/0.0 | OOD (1.000) | OOD (1.000) |

## Khuyến nghị
- Cấu hình tốt nhất (AUROC rồi F1): **tanh (bão hòa mềm), recon_w=0.6, var_w=0.4, threshold=0.5283** (AUROC 0.9947).
- Nạp `recommended.json` vào dashboard để dùng trực tiếp.