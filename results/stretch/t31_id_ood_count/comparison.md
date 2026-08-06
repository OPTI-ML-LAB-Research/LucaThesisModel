# T31 — So sánh nhận định ID/OOD: AA-only vs AAM-retrained

Cùng test set: ID = 973 phổ AA-pure (mineral≈0), OOD = 973 phổ AAM mineral-rich (mineral>0.05).

## Bảng tổng hợp

| Chỉ số | AA-only (before retrain) | AAM-retrained (after retrain) |
|---|---|---|
| Số mẫu thử | 1946 | 1946 |
| Nhận định ĐÚNG | 1886 (96.9%) | 1251 (64.3%) |
| Nhận định SAI | 60 | 695 |
| TP (OOD→OOD, đúng) | 962 | 326 |
| TN (ID→ID, đúng) | 924 | 925 |
| FP (ID→OOD, báo nhầm) | 49 | 48 |
| FN (OOD→ID, bỏ sót) | 11 | 647 |
| Accuracy | 0.9692 | 0.6429 |
| Recall / TPR | 0.9887 | 0.3350 |
| Specificity | 0.9496 | 0.9507 |
| F1 | 0.9698 | 0.4840 |
| OOD score TB (lớp ID) | 0.5470 | 0.5330 |
| OOD score TB (lớp OOD) | 0.9806 | 0.8505 |

## Diễn giải

**AA-only (before retrain):** mineral-rich là OOD thật. Model thiếu mineral references nên KHÔNG tái dựng được → OOD score cao → phát hiện đúng. Accuracy cao xác nhận model 'biết khi không biết'.

**AAM-retrained (after retrain):** model đã học mineral fingerprint nên tái dựng tốt cả phổ mineral-rich → OOD score thấp → coi mineral-rich là ID. Do đó TP giảm / FN tăng *khi xét theo nhãn OOD cũ*. Đây KHÔNG phải lỗi: nó chứng minh khi cấp thêm dữ liệu huấn luyện, mẫu trước đây là OOD chuyển thành ID — đúng tinh thần modular design.

**Kết luận cho thesis:** OOD detector hoạt động đúng (model A bắt được mẫu lạ); và framework mở rộng được (model B hấp thụ được phân phối mới). Hai kết quả bổ trợ nhau, củng cố cho thiết kế physics + OOD.