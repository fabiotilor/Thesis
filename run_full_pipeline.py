#!/usr/bin/env python3
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd
import rerun as rr
import torch

from eval_config import (
    SUBJECT_BY_CODE,
    SUBJECT_NAMES,
    DATASET_BASE_ROOT,
    VIEW_CONFIGS,
    DEFAULT_TARGET_VIEWS,
    MODEL_NAME,
    DEVICE,
    RERUN_ADDR,
    RERUN_EYE_UP,
)

import mast3r.utils.path_to_dust3r  # noqa: F401
from mast3r.model import AsymmetricMASt3R

import align_reconstruction_umeyama as baseline_mod
from align_reconstruction_umeyama import run_reconstruction as baseline_run_reconstruction

from mast3r.utils.alignment_4d import strategy1_reference, strategy2_hierarchical, strategy3_pgo
from mast3r.utils.rerun_logging import (
    init_recording,
    log_gt_sequence,
    log_aligned_sequence,
    configure_rerun_view_defaults,
)

# `4D_Umeyama.py` cannot be imported normally (module name starts with a digit),
# so we load it by file path.
import importlib.util

_umeyama4d_path = os.path.join(os.path.dirname(__file__), "4D_Umeyama.py")
_umeyama4d_spec = importlib.util.spec_from_file_location("umeyama4d", _umeyama4d_path)
_umeyama4d_mod = importlib.util.module_from_spec(_umeyama4d_spec)
assert _umeyama4d_spec and _umeyama4d_spec.loader
_umeyama4d_spec.loader.exec_module(_umeyama4d_mod)
solve_final_gt_registration = _umeyama4d_mod.solve_final_gt_registration
save_aligned_results = _umeyama4d_mod.save_aligned_results


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all subjects.")
    for code in SUBJECT_BY_CODE.keys():
        parser.add_argument(f"--{code}", action="store_true", help=f"Run subject {code}.")
    parser.add_argument(
        "--views",
        nargs="+",
        type=int,
        default=None,
        help="Optional view counts to run (default: [2,3,4] when multi-view is enabled; else [4]).",
    )
    parser.add_argument(
        "--pgo",
        action="store_true",
        help="Run only Strategy 3 (PGO) + evaluation. Baseline outputs must already exist.",
    )
    parser.add_argument(
        "--use_sam2",
        action="store_true",
        help="Use SAM2 for static mask computation instead of Farneback.",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun visualization to avoid blocking/latency.",
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


def _sorted_frame_paths(frame_dir: str):
    paths = glob.glob(os.path.join(frame_dir, "frame_*.npz"))
    if not paths:
        return []

    def _key(p):
        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]  # frame_00
        return int(stem.split("_")[1])

    return sorted(paths, key=_key)


def _write_timing_json(out_dir: str, method_label: str, n_frames: int, total_seconds: float):
    payload = {
        "strategy": method_label,
        "n_frames": int(n_frames),
        "total_seconds": float(total_seconds),
        "seconds_per_frame": float(total_seconds / max(n_frames, 1)),
    }
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _target_views_for_nviews(nviews: int):
    target_views = VIEW_CONFIGS.get(nviews)
    if target_views is None:
        target_views = DEFAULT_TARGET_VIEWS
    return target_views


def _run_eval(code: str, view_counts: list[int], use_sam2: bool = False):
    cmd = [sys.executable, "evaluate_4D.py", f"--{code}", "--views"] + [str(v) for v in view_counts]
    if use_sam2:
        cmd += ["--use_sam2"]
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

    # Load the MASt3R model once.
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[INFO] Loading model '{MODEL_NAME}' on {DEVICE} ...")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)

    cache_root = os.path.join(tempfile.gettempdir(), "mast3r_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    for subject_full, code in zip(selected_subjects, codes):
        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found, skipping: {dataset_root}")
            continue

        for nviews in view_counts:
            # Create an independent rerun recording card per (subject, view-count).
            if not args.no_rerun:
                init_recording(code, nviews)
            view_root = f"mast3r_{code}_{nviews}views"

            suffix = "_sam2" if args.use_sam2 else ""
            baseline_dir = os.path.join("aligned_outputs", "baseline" + suffix, subject_full, f"{nviews}views")
            s1_dir = os.path.join("aligned_outputs", "strategy1" + suffix, subject_full, f"{nviews}views")
            s2_dir = os.path.join("aligned_outputs", "strategy2" + suffix, subject_full, f"{nviews}views")
            s3_dir = os.path.join("aligned_outputs", "strategy3" + suffix, subject_full, f"{nviews}views")

            for d in (baseline_dir, s1_dir, s2_dir, s3_dir):
                os.makedirs(d, exist_ok=True)

            if not args.no_rerun:
                configure_rerun_view_defaults(view_root, RERUN_EYE_UP)

            frame_paths = []

            if not args.pgo:
                print(f"\n[STAGE] Baseline: subject={code} views={nviews}")
                target_views = _target_views_for_nviews(nviews)
                run_tag = view_root
                baseline_run_reconstruction(
                    model=model,
                    dataset_root=dataset_root,
                    target_views=target_views,
                    out_dir=baseline_dir,
                    cache_root=cache_root,
                    flow_threshold=1.0,
                    run_tag=run_tag,
                    skip_rerun_init=True,
                    use_sam2=args.use_sam2,
                    no_rerun=args.no_rerun,
                )

            frame_paths = _sorted_frame_paths(baseline_dir)
            if len(frame_paths) < 2:
                print(f"[WARN] Not enough baseline frames for subject={code} views={nviews}; skipping strategies.")
                continue

            # Log GT sequence only if baseline was skipped.
            # When baseline ran, it already logged:
            #   <view_root>/gt (green)
            if args.pgo and not args.no_rerun:
                try:
                    log_gt_sequence(frame_paths, log_root=view_root)
                except Exception as e:
                    print(f"[RERUN][WARN] log_gt_sequence failed for {code} {nviews}views: {e}")

            if not args.pgo:
                print(f"\n[STAGE] Strategy 1: subject={code} views={nviews}")
                s1_start = time.perf_counter()
                tf_s1 = strategy1_reference(frame_paths, dataset_root)
                s_g1, R_g1, tr_g1 = solve_final_gt_registration(frame_paths, tf_s1, dataset_root)
                save_aligned_results(
                    frame_paths,
                    tf_s1,
                    s_g1,
                    R_g1,
                    tr_g1,
                    subject_name=subject_full,
                    out_dir=s1_dir,
                    dataset_root=dataset_root,
                    method_label="strategy1",
                )
                _write_timing_json(s1_dir, "strategy1", len(frame_paths), time.perf_counter() - s1_start)
                if not args.no_rerun:
                    log_aligned_sequence(
                        frame_paths,
                        tf_s1,
                        s_g1,
                        R_g1,
                        tr_g1,
                        label="Strategy_1",
                            # Strategy1: red
                            color=[255, 0, 0],
                        dataset_root=dataset_root,
                            log_root=view_root,
                    )

                print(f"\n[STAGE] Strategy 2: subject={code} views={nviews}")
                s2_start = time.perf_counter()
                tf_s2 = strategy2_hierarchical(frame_paths, dataset_root)
                s_g2, R_g2, tr_g2 = solve_final_gt_registration(frame_paths, tf_s2, dataset_root)
                save_aligned_results(
                    frame_paths,
                    tf_s2,
                    s_g2,
                    R_g2,
                    tr_g2,
                    subject_name=subject_full,
                    out_dir=s2_dir,
                    dataset_root=dataset_root,
                    method_label="strategy2",
                )
                _write_timing_json(s2_dir, "strategy2", len(frame_paths), time.perf_counter() - s2_start)
                if not args.no_rerun:
                    log_aligned_sequence(
                        frame_paths,
                        tf_s2,
                        s_g2,
                        R_g2,
                        tr_g2,
                        label="Strategy_2",
                        # Strategy2: magenta
                        color=[255, 0, 255],
                        dataset_root=dataset_root,
                        log_root=view_root,
                    )

            print(f"\n[STAGE] Strategy 3 (PGO): subject={code} views={nviews}")
            s3_start = time.perf_counter()
            tf_s3 = strategy3_pgo(frame_paths, dataset_root, num_iters=50)
            s_g3, R_g3, tr_g3 = solve_final_gt_registration(frame_paths, tf_s3, dataset_root)
            save_aligned_results(
                frame_paths,
                tf_s3,
                s_g3,
                R_g3,
                tr_g3,
                subject_name=subject_full,
                out_dir=s3_dir,
                dataset_root=dataset_root,
                method_label="strategy3",
            )
            _write_timing_json(s3_dir, "strategy3", len(frame_paths), time.perf_counter() - s3_start)
            if not args.no_rerun:
                log_aligned_sequence(
                    frame_paths,
                    tf_s3,
                    s_g3,
                    R_g3,
                    tr_g3,
                    label="Strategy_3",
                    # Strategy3: cyan
                    color=[0, 255, 255],
                    dataset_root=dataset_root,
                    log_root=view_root,
                )

        print(f"\n[INFO] Evaluating subject {code} across methods/views ...")
        _run_eval(code, view_counts, use_sam2=args.use_sam2)

    # Aggregate results across selected subjects only.
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
    aggregated = numeric_df.groupby("strategy").mean().reset_index().sort_values(by="strategy")

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
