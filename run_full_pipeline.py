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
    DATASETS,
    get_dataset_config,
    get_subject_by_code,
    get_view_config,
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
    parser.add_argument("--data", type=str, choices=["dex-ycb", "hi4d"], default="dex-ycb", help="Dataset to use.")
    parser.add_argument("--all", action="store_true", help="Run all subjects.")
    parser.add_argument("--subjects", nargs="+", type=str, help="Specific subject codes to run.")

    # Legacy flags for DexYCB
    for code in SUBJECT_BY_CODE.keys():
        parser.add_argument(f"--{code}", action="store_true", help=f"Run subject {code} (Legacy).")

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
    return parser.parse_known_args()[0]


def _selected_subjects(args):
    dataset_type = args.data
    subj_map = get_subject_by_code(dataset_type)

    if args.all:
        codes = list(subj_map.keys())
    elif args.subjects:
        codes = args.subjects
    else:
        # Legacy flag check
        import sys
        codes = [a.lstrip('-') for a in sys.argv if a.startswith('--') and a.lstrip('-') in subj_map]
        if not codes:
            print(f"[WARN] No subject selection provided; defaulting to first subject.")
            codes = [list(subj_map.keys())[0]]

    return [subj_map[c] for c in codes], codes


def _run_eval(code: str, view_counts: list[int], dataset_type: str = "dex-ycb", no_rerun: bool = False):
    cmd = [sys.executable, "evaluate_4D.py", "--subjects", code, "--data", dataset_type, "--views"] + [
        str(v) for v in view_counts
    ]
    if no_rerun:
        cmd.append("--no-rerun")
    print(f"\nRUNNING: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = _parse_args()
    dataset_type = args.data
    dataset_config = get_dataset_config(dataset_type)
    selected_subjects, codes = _selected_subjects(args)

    if args.views is None:
        if dataset_type == "dex-ycb":
            view_counts = [2, 3, 4] if baseline_mod.RUN_MULTI_VIEW_EVAL else [4]
        else:
            # For HI4D, look at the first selected subject's pair configuration
            pair_name = selected_subjects[0].split("/")[0] if selected_subjects else "default"
            view_configs = dataset_config.get("view_configs", {})
            cfg = view_configs.get(pair_name, view_configs.get("default", {2: [], 4: []}))
            view_counts = sorted([int(k) for k in cfg.keys()])
            if not view_counts:
                view_counts = [2, 3, 4]
    else:
        view_counts = args.views

    print(f"[INFO] Dataset: {dataset_type}")
    print(f"[INFO] Selected subjects: {codes}")

    # ── Filter subjects that already have CSVs ───────────────────────────
    subjects_to_process = []
    codes_to_process = []
    for subject_full, code in zip(selected_subjects, codes):
        safe_code = code.replace("/", "_")
        csv_path = f"eval_summary_{dataset_type}_{safe_code}.csv"
        if os.path.exists(csv_path) and not args.eval_only:
            print(f"[SKIP] Subject {code} already exists ({csv_path}).")
        else:
            subjects_to_process.append(subject_full)
            codes_to_process.append(code)

    if not subjects_to_process and not args.eval_only:
        print("[INFO] All selected subjects have already been processed.")

    # ── Load VGGT4D model (skip if eval-only or nothing to process) ───────
    model = None
    if not args.eval_only and len(subjects_to_process) > 0:
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[INFO] Loading VGGT4D from '{VGGT4D_CHECKPOINT}' on {DEVICE} ...")
        model = VGGTFor4D()
        model.load_state_dict(torch.load(VGGT4D_CHECKPOINT, weights_only=True))
        model.eval()
        model = model.to(DEVICE)

    cache_root = os.path.join(tempfile.gettempdir(), "vggt4d_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: Baseline reconstruction & Stage 2: Evaluation
    # ══════════════════════════════════════════════════════════════════════
    for subject_full, code in zip(subjects_to_process, codes_to_process):
        safe_code = code.replace("/", "_")

        dataset_root = os.path.join(dataset_config["root"], subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found, skipping: {dataset_root}")
            continue

        for nviews in view_counts:
            if not args.no_rerun:
                init_recording(code, nviews)
            view_root = f"vggt4d_{dataset_type}_{safe_code}_{nviews}views"

            baseline_dir = os.path.join(
                "aligned_outputs", "baseline", dataset_type, subject_full, f"{nviews}views"
            )
            os.makedirs(baseline_dir, exist_ok=True)
            if not args.no_rerun:
                eye_up = dataset_config.get("eye_up", RERUN_EYE_UP)
                configure_rerun_view_defaults(view_root, eye_up)

            if not args.eval_only:
                print(f"\n[STAGE 1] Reconstruction: subject={code} views={nviews} dataset={dataset_type}")
                pair_name = subject_full.split("/")[0] if dataset_type == "hi4d" else None
                target_views = get_view_config(dataset_type, nviews, pair_name=pair_name)

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
                    dataset_type=dataset_type,
                )

        # ══════════════════════════════════════════════════════════════════
        print(f"\n[STAGE 2] Evaluating subject {code} ...")
        _run_eval(code, view_counts, dataset_type=dataset_type, no_rerun=args.no_rerun)

    # ── Aggregate results across selected subjects ───────────────────────
    csv_files = []
    for code in codes:
        safe_code = code.replace("/", "_")
        csv_path = f"eval_summary_{dataset_type}_{safe_code}.csv"
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
    print(f"CROSS-SUBJECT AGGREGATED RESULTS - {dataset_type.upper()} (MEAN ACROSS PROCESSED SUBJECTS)")
    print("=" * 80)

    pd.set_option("display.precision", 5)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", None)
    cols_to_show = [
        "strategy", "n_frames", "chamfer", "chamfer_4d", "delta_consistency",
        "completeness", "accuracy",
        "static_comp", "dyn_comp", "static_acc", "dyn_acc", "motion_gap",
        "ate", "rpe", "rot_error", "focal_error", "pp_error",
        "jitter_mean", "jitter_std", "jitter_p95", "jitter_max",
        "drift_mean", "hf_jitter"
    ]
    cols_to_show = [c for c in cols_to_show if c in aggregated.columns]
    print(aggregated[cols_to_show].to_string(index=False))

    out_file = f"eval_summary_{dataset_type}_ALL_SUBJECTS.csv"
    aggregated.to_csv(out_file, index=False)
    print(f"\n[INFO] Aggregated results saved to {out_file}")


if __name__ == "__main__":
    main()
