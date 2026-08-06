"""Collect post-training summary for the handover document.

Reads results/training_log.csv + checkpoints/best.pt and prints a
Markdown-formatted summary the user can paste into the [FILL AFTER RUN]
sections of CHAT3_CORE_ENGINE_HANDOVER.md.

Usage:
    python scripts/collect_run_summary.py
    python scripts/collect_run_summary.py --csv results/training_log.csv \
                                          --ckpt checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def gate_verdict(best_mae: float) -> str:
    if best_mae <= 0.020:
        return "PASS-target"
    if best_mae <= 0.025:
        return "PASS-floor"
    if best_mae <= 0.040:
        return "BORDERLINE"
    return "FAIL"


def gate_reason(best_mae: float) -> str:
    if best_mae <= 0.020:
        return f"Best val MAE {best_mae:.4f} ≤ target 0.020. Proceed to Phase 3."
    if best_mae <= 0.025:
        return (f"Best val MAE {best_mae:.4f} > target 0.020 but ≤ floor 0.025. "
                f"Proceed to Phase 3; document MAE as known limitation.")
    if best_mae <= 0.040:
        return (f"Best val MAE {best_mae:.4f} > floor 0.025 but ≤ hard floor 0.040. "
                f"Apply 1-day retune (lower lr, disable augmentation, or resume +30 ep) "
                f"per TRAINING_GUIDE §3.2 before re-evaluating.")
    return (f"Best val MAE {best_mae:.4f} > hard floor 0.040. Stop normal flow; "
            f"apply fallback (drop physics loss, retrain pure regressor) per "
            f"TRAINING_GUIDE §6.")


def env_block() -> str:
    """Best-effort environment summary (works without torch installed)."""
    lines = [
        f"- **Date / time:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Machine:** {platform.platform()}",
        f"- **Python:** {sys.version.split()[0]}",
    ]
    try:
        import torch
        lines.append(f"- **PyTorch:** {torch.__version__}")
        lines.append(
            f"- **Device used:** "
            f"{'cuda:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
        )
    except ImportError:
        lines.append("- **PyTorch:** (not importable in this environment)")
        lines.append("- **Device used:** [FILL — run from torch env]")
    return "\n".join(lines)


def summarise_csv(rows: List[Dict[str, str]]) -> Dict[str, object]:
    """Extract the headline numbers."""
    if not rows:
        raise ValueError("Empty CSV")

    val_mae_pairs: List[tuple] = []
    for r in rows:
        ep = safe_float(r.get("epoch"))
        v = safe_float(r.get("val_mae"))
        if ep is not None and v is not None:
            val_mae_pairs.append((int(ep), v))
    if not val_mae_pairs:
        raise ValueError("No valid (epoch, val_mae) rows found")

    best_epoch, best_mae = min(val_mae_pairs, key=lambda kv: kv[1])
    best_row = next(r for r in rows if int(safe_float(r["epoch"])) == best_epoch)
    final_row = rows[-1]
    final_epoch = int(safe_float(final_row["epoch"]))
    n_epochs = len(rows)

    total_seconds = sum(
        s for s in (safe_float(r.get("epoch_seconds")) for r in rows) if s
    )

    train_mae_at_best = safe_float(best_row.get("train_mae"))
    val_mae_at_best = safe_float(best_row.get("val_mae"))
    gap = (
        val_mae_at_best - train_mae_at_best
        if (train_mae_at_best is not None and val_mae_at_best is not None)
        else None
    )

    physics_at_best = safe_float(best_row.get("val_loss_physics"))

    # Diagnostic: physics/quant ratio at epoch 10 (P1-4 / P2-7).
    ratio_ep10 = None
    for r in rows:
        if int(safe_float(r["epoch"])) == 10:
            q = safe_float(r.get("train_loss_quant"))
            p = safe_float(r.get("train_loss_physics"))
            if q and q > 1e-9 and p is not None:
                ratio_ep10 = p / q
            break

    early_stopped = (n_epochs - 1) < final_epoch  # rare; CSV epoch jumped (resume)
    # Better heuristic: did we train for fewer than the configured epochs?
    # That info isn't in the CSV alone -- mark it as unknown.

    return {
        "best_mae": best_mae,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "n_epochs_trained": n_epochs,
        "total_seconds": total_seconds,
        "train_mae_at_best": train_mae_at_best,
        "val_mae_at_best": val_mae_at_best,
        "gap_at_best": gap,
        "physics_at_best": physics_at_best,
        "physics_quant_ratio_ep10": ratio_ep10,
    }


def fmt_optional(v: Optional[float], spec: str = ".4f") -> str:
    if v is None:
        return "[N/A]"
    return f"{v:{spec}}"


def render_markdown(
    csv_path: Path, ckpt_info: Dict[str, object], summary: Dict[str, object],
    head_rows: List[Dict[str, str]], tail_rows: List[Dict[str, str]],
) -> str:
    """Build the paste-ready Markdown block."""
    best = summary["best_mae"]
    verdict = gate_verdict(best)
    reason = gate_reason(best)

    csv_keys = [
        "epoch", "lr", "train_loss_total", "train_loss_quant",
        "train_loss_physics", "train_mae",
        "val_loss_total", "val_loss_quant", "val_loss_physics",
        "val_mae", "best_val_mae", "epoch_seconds",
    ]

    def csv_row_str(r: Dict[str, str]) -> str:
        return ",".join(r.get(k, "") for k in csv_keys)

    rows_text = "\n".join(csv_row_str(r) for r in head_rows + tail_rows)

    out = [
        "<!-- ============ PASTE THIS INTO CHAT3_CORE_ENGINE_HANDOVER.md ============ -->",
        "",
        "### D.1 Environment",
        "",
        env_block(),
        "",
        "### D.3 Headline numbers",
        "",
        "| Metric | Value | Status vs target |",
        "|---|---|---|",
        f"| Best val MAE | **{best:.4f}** | target ≤ 0.020 / floor ≤ 0.025 → **{verdict}** |",
        f"| Best epoch | {summary['best_epoch']} | (which epoch saved `best.pt`) |",
        f"| Final epoch trained | {summary['final_epoch']} | of {summary['n_epochs_trained']} total epochs in log |",
        f"| Total wall time | {summary['total_seconds']/60:.1f} min | (sum of `epoch_seconds`) |",
        f"| Train MAE at best epoch | {fmt_optional(summary['train_mae_at_best'])} | |",
        f"| Train/val MAE gap at best | {fmt_optional(summary['gap_at_best'])} | overfitting indicator (positive = val worse) |",
        f"| Val physics loss at best | {fmt_optional(summary['physics_at_best'])} | reconstruction quality |",
        f"| physics/quant ratio @ ep10 | {fmt_optional(summary['physics_quant_ratio_ep10'], '.2f')} | P2-7 diagnostic; > 5 means physics dominates |",
        "",
        f"**Checkpoint metadata:** epoch={ckpt_info.get('epoch', '[N/A]')}, "
        f"val_mae={ckpt_info.get('val_mae', '[N/A]')}",
        "",
        "### D.4 Training log CSV — first / last 5 rows",
        "",
        "```csv",
        ",".join(csv_keys),
        rows_text,
        "```",
        "",
        "### D.6 Day-7 gate decision",
        "",
        f"**Decision:** **{verdict}**",
        "",
        f"**Rationale:** {reason}",
        "",
        "<!-- ====== Manually fill the remaining [FILL AFTER RUN] sections (D.2, D.5, D.7, P1-4, P2-2, P2-3, P2-7, B.3, C-extra) ====== -->",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/training_log.csv")
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--head", type=int, default=5,
                   help="How many rows to show from the start (default 5)")
    p.add_argument("--tail", type=int, default=5,
                   help="How many rows to show from the end (default 5)")
    args = p.parse_args()

    rows = load_csv(Path(args.csv))
    summary = summarise_csv(rows)

    ckpt_info: Dict[str, object] = {}
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        try:
            import torch
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_info = {
                "epoch": ck.get("epoch"),
                "val_mae": ck.get("val_metrics", {}).get("val_mae"),
            }
        except ImportError:
            print("# WARNING: torch not importable; skipping checkpoint inspection",
                  file=sys.stderr)
        except Exception as e:
            print(f"# WARNING: failed to load checkpoint: {e}", file=sys.stderr)
    else:
        print(f"# NOTE: {ckpt_path} not found; skipping checkpoint metadata",
              file=sys.stderr)

    head_rows = rows[:args.head]
    tail_rows = rows[-args.tail:] if len(rows) > args.head else []

    print(render_markdown(Path(args.csv), ckpt_info, summary,
                          head_rows, tail_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
