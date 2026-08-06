"""Stretch Master Orchestrator — chạy toàn bộ stretch tasks theo thứ tự ưu tiên.

Pipeline:
    Tier 1 (critical, hoàn thiện thesis):
        T30A   Real MoS2 demo                              (~5 min)
        T29A   AAM data prep (one-time)                    (~5-10 min)
        T29B   AAM zero-shot test (BEFORE)                 (~10 min)
        T29D   AAM retrain (7 outputs)                     (~60-120 min)
        T29E   AAM post-retrain test (AFTER)               (~5-10 min)

    Tier 2 (bonus):
        T31    API pharmaceutical zero-shot OOD            (~15-20 min)
        T32    Bacteria-ID cross-instrument zero-shot OOD  (~20-30 min)

Mỗi task có thể skip qua flag. Output luôn được save sau từng task -- 
nếu task sau crash, không mất kết quả task trước.

Usage:
    # Chạy đầy đủ (~2-4h tổng)
    python scripts/stretch/run_all_stretch.py

    # Smoke test (chạy với data ít, epochs ít)
    python scripts/stretch/run_all_stretch.py --smoke

    # Chỉ Tier 1
    python scripts/stretch/run_all_stretch.py --skip-tier2

    # Skip retrain (giữ AA-only baseline)
    python scripts/stretch/run_all_stretch.py --skip-retrain

    # Resume sau crash: skip task đã chạy xong
    python scripts/stretch/run_all_stretch.py --skip-t30a --skip-t29a
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(label: str, cmd: list[str], dry_run: bool = False) -> bool:
    """Run a single task. Returns True on success."""
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"  cmd: {' '.join(cmd)}")
    if dry_run:
        print("  [DRY-RUN] skipped")
        return True
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  [FAIL] {label} returned {result.returncode} after {elapsed:.0f}s")
        return False
    print(f"\n  [OK] {label} done in {elapsed:.0f}s")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Skips
    p.add_argument("--skip-t30a", action="store_true", help="Skip MoS2 demo")
    p.add_argument("--skip-t29a", action="store_true", help="Skip AAM prep")
    p.add_argument("--skip-t29b", action="store_true", help="Skip AAM zero-shot")
    p.add_argument("--skip-retrain", action="store_true", help="Skip T29D + T29E")
    p.add_argument("--skip-t29d", action="store_true", help="Skip retrain only")
    p.add_argument("--skip-t29e", action="store_true", help="Skip post-train test")
    p.add_argument("--skip-t31", action="store_true", help="Skip API OOD")
    p.add_argument("--skip-t32", action="store_true", help="Skip bacteria OOD")
    p.add_argument("--skip-tier2", action="store_true", help="Skip T31 + T32")
    # Speed knobs
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: small n, 2 epochs")
    p.add_argument("--mc", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    # Continue on failure or abort?
    p.add_argument("--continue-on-fail", action="store_true",
                   help="Continue even if a step fails")
    args = p.parse_args()

    PY = sys.executable
    SCRIPT_DIR = Path(__file__).parent
    results = {}

    def step(name, label, cmd):
        if args.continue_on_fail:
            ok = run_step(label, cmd, dry_run=args.dry_run)
        else:
            ok = run_step(label, cmd, dry_run=args.dry_run)
            if not ok:
                print(f"\n[STOP] task {name} failed, aborting.")
                print("       Re-run with --continue-on-fail to keep going.")
                sys.exit(1)
        results[name] = ok
        return ok

    t_start = time.time()
    print(f"\n{'#' * 70}")
    print(f"#  STRETCH PIPELINE  (smoke={args.smoke})")
    print(f"{'#' * 70}")

    # ---------------- TIER 1: CRITICAL ----------------

    # T30A — Real MoS2 demo
    if not args.skip_t30a:
        cmd = [PY, str(SCRIPT_DIR / "run_t30a_mos2.py"),
                "--mc", str(args.mc if not args.smoke else 20)]
        step("T30A", "T30A — Real MoS2 Demo", cmd)

    # T29A — AAM data prep (one-time, idempotent: skip if cache exists)
    aam_cache = Path("data/processed/aam/spectra.pt")
    if args.skip_t29a:
        print(f"\n[SKIP] T29A (user flag)")
    elif aam_cache.exists():
        print(f"\n[SKIP] T29A (cache already exists at {aam_cache})")
    else:
        cmd = [PY, str(SCRIPT_DIR / "t29a_prepare_aam.py")]
        if args.smoke:
            cmd += ["--n-limit", "1000"]
        step("T29A", "T29A — AAM Data Prep", cmd)

    # T29B — AAM zero-shot (BEFORE retrain)
    if not args.skip_t29b:
        cmd = [PY, str(SCRIPT_DIR / "run_t29b_zeroshot_pre.py"),
                "--mc", str(args.mc if not args.smoke else 15)]
        if args.smoke:
            cmd += ["--max-samples", "100"]
        step("T29B", "T29B — AAM Zero-shot Test (BEFORE)", cmd)

    # T29D — Retrain model
    if not args.skip_retrain and not args.skip_t29d:
        cmd = [PY, str(SCRIPT_DIR / "run_t29d_retrain_aam.py")]
        if args.smoke:
            cmd += ["--smoke"]
        else:
            cmd += ["--epochs", "50"]
        step("T29D", "T29D — AAM Retrain (7 outputs)", cmd)

    # T29E — Post-retrain test (AFTER)
    if not args.skip_retrain and not args.skip_t29e:
        cmd = [PY, str(SCRIPT_DIR / "run_t29e_posttrain.py")]
        if args.smoke:
            cmd += ["--max-samples", "100"]
        step("T29E", "T29E — AAM Post-train Test (AFTER)", cmd)

    # ---------------- TIER 2: BONUS ----------------

    if not args.skip_tier2:
        # T31 — API
        if not args.skip_t31:
            cmd = [PY, str(SCRIPT_DIR / "run_t31_api.py")]
            if args.smoke:
                cmd += ["--n-api", "100", "--n-id-max", "100"]
            step("T31", "T31 — API Cross-domain OOD", cmd)

        # T32 — Bacteria
        if not args.skip_t32:
            cmd = [PY, str(SCRIPT_DIR / "run_t32_bacteria.py")]
            if args.smoke:
                cmd += ["--n-bacteria", "100", "--n-id-max", "100"]
            step("T32", "T32 — Bacteria Cross-instrument OOD", cmd)
    else:
        print("\n[SKIP] Tier 2 (T31 + T32) by user flag")

    # ---------------- Summary ----------------
    total_elapsed = time.time() - t_start
    print("\n" + "#" * 70)
    print("#  STRETCH PIPELINE SUMMARY")
    print("#" * 70)
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {name}: {status}")
    print(f"\n  Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"\n  Handover files (paste these to report-writing chat):")
    handovers = [
        "results/stretch/t30a_mos2_handover.md",
        "results/stretch/t29b_handover.md",
        "results/stretch/t29e_handover.md",
        "results/stretch/t29_paired_comparison.md",
        "results/stretch/t31_api_handover.md",
        "results/stretch/t32_bacteria_handover.md",
        "results/stretch/STRETCH_MASTER_HANDOVER.md",  # to be generated next
    ]
    for h in handovers:
        if Path(h).exists():
            print(f"    ✓ {h}")
        else:
            print(f"    ✗ {h} (not generated)")

    # Try generating master handover
    master_script = SCRIPT_DIR / "build_master_handover.py"
    if master_script.exists() and not args.dry_run:
        print("\n  Building master handover ...")
        subprocess.run([PY, str(master_script)])


if __name__ == "__main__":
    main()
