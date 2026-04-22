#!/usr/bin/env python3
"""
VGGT4D Full Evaluation Pipeline

Executes two stages:
  1. Baseline reconstruction  (align_reconstruction_umeyama.py)
  2. Native 4D evaluation     (evaluate_4D.py)

Post-hoc alignment strategies (strategy1/2/3) are removed — VGGT4D provides
native temporal consistency via cross-view spatio-temporal attention.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import pandas as pd
import torch

from eval_config import (
    SUBJECT_BY_CODE,
    SUBJECT_NAMES,
    DATASET_BASE_ROOT,
    VIEW_CONFIGS,
    DEFAULT_TARGET_VIEWS,
    VGGT4D_CHECKPOINT,
    DEVICE,
    RERUN_EYE_UP,
)

# ── VGGT4D model ──────────────────────────────────────────────────────────────
from vggt4d.models.vggt4d import VGGTFor4D

# ── Reconstruction ────────────────────────────────────────────────────────────
import align_reconstruction_umeyama as baseline_mod
from align_reconstruction_umeyama import run_reconstruction as baseline_run_reconstruction

from vggt.utils.rerun_logging import (
    init_recording,
    configure_rerun_view_defaults,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="VGGT4D pipeline: reconstruction → native 4D evaluation"
    )
    parser.add_argument("--all", action="store_true", help="Run all subjects.")
    for code in SUBJECT_BY_CODE.keys():
        parser.add_argument(f"--{code}", action="store_true", help=f"Run subject {code}.")
    parser.add_argument(
        "--views",
        nargs="+",
        type=int,
        default=None,
        help="View counts to run (default: [2,3,4] when multi-view is enabled; else [4]).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip reconstruction; only run evaluation on existing outputs.",
    )
    parser.add_argument(
        "--all-at-once",
        action="store_true",
        help="Attempt to process the entire sequence in a single batch (stressed VRAM test).",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun logging to prevent blocking when data channel is saturated.",
    )
    return parser.parse_args()


def _selected_subjects(args):
    if args.all:
        codes = list(SUBJECT_BY_CODE.keys())
    else:
        codes = [code for code in SUBJECT_BY_CODE.keys() if getattr(args, code)]
    if not codes:
        print("[WARN] No subject selection flag provided; defaulting to --01")
        codes = ["01"]
    return [SUBJECT_BY_CODE[c] for c in codes], codes


def _target_views_for_nviews(nviews: int):
    target_views = VIEW_CONFIGS.get(nviews)
    if target_views is None:
        target_views = DEFAULT_TARGET_VIEWS
    return target_views


def _run_eval(code: str, view_counts: list[int], no_rerun: bool = False):
    cmd = [sys.executable, "evaluate_4D.py", f"--{code}", "--views"] + [
        str(v) for v in view_counts
    ]
    if no_rerun:
        cmd.append("--no-rerun")
    print(f"\nRUNNING: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = _parse_args()
    selected_subjects, codes = _selected_subjects(args)

    if args.views is None:
        view_counts = [2, 3, 4] if baseline_mod.RUN_MULTI_VIEW_EVAL else [4]
    else:
        view_counts = args.views

    print(f"[INFO] Selected subjects: {codes}")

    # ── Load VGGT4D model (skip if eval-only) ────────────────────────────
    model = None
    if not args.eval_only:
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[INFO] Loading VGGT4D from '{VGGT4D_CHECKPOINT}' on {DEVICE} ...")
        model = VGGTFor4D()
        model.load_state_dict(torch.load(VGGT4D_CHECKPOINT, weights_only=True))
        model.eval()
        model = model.to(DEVICE)

    cache_root = os.path.join(tempfile.gettempdir(), "vggt4d_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: Baseline reconstruction
    # ══════════════════════════════════════════════════════════════════════
    for subject_full, code in zip(selected_subjects, codes):
        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found, skipping: {dataset_root}")
            continue

        for nviews in view_counts:
            if not args.no_rerun:
                init_recording(code, nviews)
            view_root = f"vggt4d_{code}_{nviews}views"

            baseline_dir = os.path.join(
                "aligned_outputs", "baseline", subject_full, f"{nviews}views"
            )
            os.makedirs(baseline_dir, exist_ok=True)
            if not args.no_rerun:
                configure_rerun_view_defaults(view_root, RERUN_EYE_UP)

            if not args.eval_only:
                print(f"\n[STAGE 1] Reconstruction: subject={code} views={nviews}")
                target_views = _target_views_for_nviews(nviews)
                baseline_run_reconstruction(
                    model=model,
                    dataset_root=dataset_root,
                    target_views=target_views,
                    out_dir=baseline_dir,
                    cache_root=cache_root,
                    flow_threshold=1.0,
                    run_tag=view_root,
                    skip_rerun_init=True,
                    all_at_once=args.all_at_once,
                    no_rerun=args.no_rerun,
                )

        # ══════════════════════════════════════════════════════════════════
        print(f"\n[STAGE 2] Evaluating subject {code} ...")
        _run_eval(code, view_counts, no_rerun=args.no_rerun)

    # ── Aggregate results across selected subjects ───────────────────────
    csv_files = []
    for code in codes:
        csv_path = f"eval_summary_{code}.csv"
        if os.path.exists(csv_path):
            csv_files.append(csv_path)

    if not csv_files:
        print("[WARN] No per-subject CSV files found to aggregate.")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    if combined_df.empty:
        print("[WARN] Combined CSV is empty.")
        return

    numeric_df = combined_df.select_dtypes(include="number")
    numeric_df["strategy"] = combined_df["strategy"]
    aggregated = (
        numeric_df.groupby("strategy").mean().reset_index().sort_values(by="strategy")
    )

    print("\n\n" + "=" * 80)
    print("CROSS-SUBJECT AGGREGATED RESULTS (MEAN ACROSS PROCESSED SUBJECTS)")
    print("=" * 80)

    pd.set_option("display.precision", 5)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", None)
    print(aggregated.to_string(index=False))

    out_file = "eval_summary_ALL_SUBJECTS.csv"
    aggregated.to_csv(out_file, index=False)
    print(f"\n[INFO] Aggregated results saved to {out_file}")


if __name__ == "__main__":
    main()
