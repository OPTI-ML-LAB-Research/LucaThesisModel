"""T25 — End-to-end benchmark facade.

Convenience CLI that:
  1. (optionally) Fits the PCA+SVM baseline       → ``scripts.train_baseline_pca_svm``
  2. (optionally) Trains the ResNet-only baseline → ``scripts.train_baseline_resnet_only``
  3. Always runs the comparison                   → ``src.eval.compare``

In a typical Phase B run, you'd call the underlying CLIs directly (more
control), but this entry exists for the one-line reproduction story used
in ``scripts/reproduce_results.sh`` (Day 14 deliverable).

Examples
--------
::

    # Step-by-step (recommended)
    python -m src.models.baselines.pca_svm
    python -m src.models.baselines.resnet_only
    python -m src.eval.compare

    # One-liner via this facade (T25-only — assumes baselines already exist)
    python -m src.eval.benchmark --compare-only

    # End-to-end (will fit baselines if missing)
    python -m src.eval.benchmark
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], dry: bool = False) -> int:
    print(f"  $ {' '.join(cmd)}")
    if dry:
        return 0
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip training; just run T25 comparison.")
    parser.add_argument("--skip-pca-svm", action="store_true")
    parser.add_argument("--skip-resnet", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pca_ckpt = Path("checkpoints/baselines/pca_svm.pkl")
    rn_ckpt = Path("checkpoints/baselines/resnet_only_best.pt")

    if not args.compare_only:
        if not args.skip_pca_svm and not pca_ckpt.exists():
            print("[1/3] Fitting PCA+SVM ...")
            rc = _run([sys.executable, "-m", "src.models.baselines.pca_svm"],
                      dry=args.dry_run)
            if rc != 0:
                return rc
        else:
            print(f"[1/3] PCA+SVM: skip (exists={pca_ckpt.exists()})")

        if not args.skip_resnet and not rn_ckpt.exists():
            print("[2/3] Training ResNet-only ...")
            rc = _run([sys.executable, "-m", "src.models.baselines.resnet_only",
                       "--device", args.device], dry=args.dry_run)
            if rc != 0:
                return rc
        else:
            print(f"[2/3] ResNet-only: skip (exists={rn_ckpt.exists()})")

    print("[3/3] Running comparison ...")
    cmd = [sys.executable, "-m", "src.eval.compare", "--device", args.device]
    if args.skip_resnet:
        cmd.append("--skip-resnet")
    return _run(cmd, dry=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
