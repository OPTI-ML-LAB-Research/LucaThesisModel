"""dashboard/ood_controls.py — điều chỉnh OOD score trực tiếp trong dashboard.

Module độc lập, KHÔNG sửa src/inference/ood.py hay predict.py. Nó cho phép:

  1. Đổi trọng số (recon_weight / var_weight) — kể cả 1.0/0.0 (physics-only).
  2. Đổi threshold thủ công, hoặc nạp threshold đã calibrate theo F1.
  3. Thử các thang chuẩn hóa khác nhau (clip / no-clip / tanh / rank).

CÁCH HOẠT ĐỘNG
--------------
OOD score gốc của repo = w_r · norm(recon_err) + w_v · norm(var), với
norm(x) = min(x / x_p95, 1.0). Vấn đề: norm bị CLIP ở 1.0 nên physics "mất
tiếng nói" khi recon_err đã vượt p95 (không phân biệt cosine 0.5 với 0.13).

Để đổi trọng số mà KHÔNG phải chạy lại model, ta cần hai thành phần THÔ:
recon_err_raw (= 1 - cosine) và var_raw (predictive variance). May mắn là
``OODScorer.score_batch(..., return_components=True)`` đã trả về chúng. Ta
chuẩn hóa lại từ các giá trị thô này theo trọng số/thang do người dùng chọn.

Hàm chính: ``rescore(recon_raw, var_raw, cal, *, recon_w, var_w, scale, threshold)``
trả về (score, is_ood) — thuần numpy, tức thời, không cần forward lại.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ───────────────────────── các thang chuẩn hóa ──────────────────────────────

def _norm_clip(x, p95):
    """Thang gốc của repo: min(x / p95, 1.0). Bị chặn trên ở 1.0."""
    return np.minimum(x / (p95 + 1e-12), 1.0)


def _norm_noclip(x, p95):
    """Như trên nhưng KHÔNG clip: physics far-OOD vẫn vượt 1.0 -> giữ độ phân
    giải ở vùng cực lạ (MoS2 sẽ tách hẳn khỏi AAM thay vì cùng đụng trần)."""
    return x / (p95 + 1e-12)


def _norm_tanh(x, p95):
    """tanh(x / p95): mượt, bão hòa mềm về 1.0, không có ngưỡng gãy cứng.
    Nhạy hơn ở vùng giữa, nén nhẹ ở đuôi."""
    return np.tanh(x / (p95 + 1e-12))


SCALES = {
    "clip (gốc repo)": _norm_clip,
    "no-clip (giữ độ phân giải far-OOD)": _norm_noclip,
    "tanh (bão hòa mềm)": _norm_tanh,
}


# ───────────────────────────── rescoring ────────────────────────────────────

@dataclass
class CalStats:
    """Thống kê calibration tối thiểu cần để chuẩn hóa lại."""
    recon_p95: float
    var_p95: float
    score_p95: float   # threshold gốc của repo (quantile-95 trên val)


def rescore(recon_raw, var_raw, cal: CalStats, *,
            recon_w: float = 0.6, var_w: float = 0.4,
            scale: str = "clip (gốc repo)",
            threshold: float | None = None):
    """Tính lại OOD score + nhãn từ thành phần thô, theo cấu hình tùy chỉnh.

    Args:
        recon_raw: mảng (N,) hoặc scalar — reconstruction error thô (= 1 - cosine).
        var_raw:   mảng (N,) hoặc scalar — predictive variance thô.
        cal:       CalStats (recon_p95, var_p95, score_p95).
        recon_w, var_w: trọng số. Tự chuẩn hóa về tổng 1. (1.0, 0.0) = physics-only.
        scale:     khóa trong SCALES.
        threshold: ngưỡng thủ công; None = dùng cal.score_p95.

    Returns:
        (score, is_ood): cùng shape với đầu vào.
    """
    recon_raw = np.asarray(recon_raw, dtype=np.float64)
    var_raw = np.asarray(var_raw, dtype=np.float64)

    s = recon_w + var_w
    if s <= 0:
        raise ValueError("Tổng trọng số phải > 0.")
    recon_w, var_w = recon_w / s, var_w / s   # chuẩn hóa về tổng 1

    fn = SCALES[scale]
    recon_n = fn(recon_raw, cal.recon_p95)
    var_n = fn(var_raw, cal.var_p95)
    score = recon_w * recon_n + var_w * var_n

    thr = cal.score_p95 if threshold is None else threshold
    is_ood = score > thr
    return score, is_ood


# ──────────────────── calibrate threshold theo F1 (giải H2) ─────────────────

def best_f1_threshold(scores_id, scores_ood, n_grid: int = 200):
    """Tìm threshold tối ưu F1 trên một tập có nhãn ID/OOD rõ ràng.

    Đây là cách hiệu chỉnh ngưỡng ĐÚNG (thay cho quantile-95 mù trên Split A,
    vốn bị lệch do composition-shift — chính là hạn chế H2 trong báo cáo).

    Args:
        scores_id:  (N_id,)  OOD score của mẫu ID thật.
        scores_ood: (N_ood,) OOD score của mẫu OOD thật.
    Returns:
        dict: best_threshold, f1, precision, recall, tpr, fpr, và đường cong.
    """
    scores_id = np.asarray(scores_id, dtype=np.float64)
    scores_ood = np.asarray(scores_ood, dtype=np.float64)
    lo = min(scores_id.min(), scores_ood.min())
    hi = max(scores_id.max(), scores_ood.max())
    grid = np.linspace(lo, hi, n_grid)

    best = {"f1": -1.0}
    curve = []
    eps = 1e-12
    for t in grid:
        tp = float((scores_ood > t).sum())
        fn = float((scores_ood <= t).sum())
        fp = float((scores_id > t).sum())
        tn = float((scores_id <= t).sum())
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        curve.append((float(t), f1, prec, rec))
        if f1 > best["f1"]:
            best = {
                "best_threshold": float(t), "f1": f1,
                "precision": prec, "recall": rec,
                "tpr": rec, "fpr": fp / (fp + tn + eps),
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            }
    best["curve"] = curve
    return best
