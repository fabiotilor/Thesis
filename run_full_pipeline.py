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
    DATASETS,
    DEVICE,
    RERUN_ADDR,
    RERUN_EYE_UP,
)

import align_reconstruction_umeyama as baseline_mod
from align_reconstruction_umeyama import run_reconstruction as baseline_run_reconstruction

from pi3.utils.alignment_4d import strategy1_reference, strategy2_hierarchical, strategy3_pgo
from pi3.utils.rerun_logging import (
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
_umeyama4d_mod.solve_final_gt_registration = getattr(_umeyama4d_mod, "solve_final_gt_registration", None)  # Safe load
_umeyama4d_spec.loader.exec_module(_umeyama4d_mod)
solve_final_gt_registration = _umeyama4d_mod.solve_final_gt_registration
save_aligned_results = _umeyama4d_mod.save_aligned_results


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=["dex-ycb", "hi4d"], default="dex-ycb", help="Dataset to use")
    parser.add_argument("--all", action="store_true", help="Run all subjects.")
    parser.add_argument("--subjects", nargs="+", type=str, help="Specific subject codes to run.")
    parser.add_argument("--model", type=str, choices=["pi3", "pi3x"], default="pi3", help="Model to evaluate")
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
        "--no-rerun",
        action="store_true",
        help="Disable Rerun visualization to avoid blocking/latency.",
    )
    parser.add_argument("--limit-frames", type=int, default=25, help="Limit number of frames to process")
    parser.add_argument(
        "--opt",
        action="store_true",
        help="Run temporal optimization on base strategy frames and evaluate exclusively.",
    )
    parser.add_argument(
        "--opt-base",
        type=str,
        default="strategy2",
        help="Base alignment strategy to smooth (default: strategy2).",
    )
    parser.add_argument(
        "--opt-sigma",
        type=float,
        default=4.0,
        help="Gaussian temporal window size for smoothing.",
    )
    parser.add_argument(
        "--opt-alpha",
        type=float,
        default=1.0,
        help="Blending factor for smoothing (0.0=original, 1.0=fully smoothed).",
    )

    # Keep backward compatibility for subject flags if needed, or just use --subjects
    return parser.parse_known_args()[0]


def _selected_subjects(args, dataset_config):
    subject_names = dataset_config["subject_names"]
    subject_by_code = {name.split("subject-")[1][:2] if "subject-" in name else name: name for name in subject_names}

    if args.all:
        codes = list(subject_by_code.keys())
    elif args.subjects:
        codes = args.subjects
    else:
        # Check if any legacy flags were used (e.g., --01)
        # This is a bit hacky since we used parse_known_args
        import sys
        codes = [a.lstrip('-') for a in sys.argv if a.startswith('--') and a.lstrip('-') in subject_by_code]
        if not codes:
            print(f"[WARN] No subject selection provided; defaulting to first subject.")
            codes = [list(subject_by_code.keys())[0]]

    return [subject_by_code[c] for c in codes], codes


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


def _target_views_for_nviews(nviews: int, dataset_config, subject_name=None):
    view_configs = dataset_config.get("view_configs", {})

    # Try pair-specific config
    if subject_name:
        pair_prefix = subject_name.split("/")[0]
        if pair_prefix in view_configs:
            target_views = view_configs[pair_prefix].get(nviews)
            if target_views:
                return target_views

    # Fallback to default or flat config
    default_config = view_configs.get("default", view_configs)
    target_views = default_config.get(nviews)

    if target_views is None:
        target_views = dataset_config.get("default_target_views")
    return target_views


def _run_eval(code: str, view_counts: list[int], dataset_type: str, model_type: str, opt: bool = False):
    cmd = [sys.executable, "evaluate_4D.py", "--subjects", code, "--data", dataset_type, "--model", model_type,
           "--views"] + [str(v) for v in view_counts]
    if opt:
        cmd.append("--opt")
    print(f"\nRUNNING: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = _parse_args()
    dataset_config = DATASETS[args.data]
    dataset_type = args.data
    selected_subjects, codes = _selected_subjects(args, dataset_config)

    if args.views is None:
        view_counts = [2, 3, 4] if baseline_mod.RUN_MULTI_VIEW_EVAL else [4]
        if dataset_type == "hi4d":
            # Override for Hi4D if needed, or use configs
            view_counts = [v for v in dataset_config.get("view_configs", {}).keys()]
    else:
        view_counts = args.views

    print(f"[INFO] Selected subjects: {codes}")
    print(f"[INFO] Dataset: {dataset_type}")

    # Load the Pi3/Pi3X model once.
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[INFO] Loading model '{args.model}' on {DEVICE} ...")
    if args.model == "pi3":
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained("yyfz233/Pi3").to(DEVICE).eval()
    elif args.model == "pi3x":
        from pi3.models.pi3x import Pi3X
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        model.disable_multimodal()
        model = model.to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    cache_root = os.path.join(tempfile.gettempdir(), f"pi3_alignment_cache_{dataset_type}")
    os.makedirs(cache_root, exist_ok=True)

    for subject_full, code in zip(selected_subjects, codes):
        safe_code = code.replace("/", "_")
        csv_path = f"eval_summary_{args.model}_{dataset_type}_{safe_code}.csv"
        if os.path.exists(csv_path):
            print(f"[INFO] Skipping subject {code} as evaluation results already exist: {csv_path}")
            continue

        dataset_root = os.path.join(dataset_config["root"], subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found, skipping: {dataset_root}")
            continue

        for nviews in view_counts:
            # Create an independent rerun recording card per (subject, view-count).
            if not args.no_rerun:
                init_recording(code, nviews)
            view_root = f"{args.model}_{dataset_type}_{code}_{nviews}views"

            baseline_dir = os.path.join("aligned_outputs", args.model, dataset_type, "baseline", subject_full,
                                        f"{nviews}views")
            s1_dir = os.path.join("aligned_outputs", args.model, dataset_type, "strategy1", subject_full,
                                  f"{nviews}views")
            s2_dir = os.path.join("aligned_outputs", args.model, dataset_type, "strategy2", subject_full,
                                  f"{nviews}views")
            s3_dir = os.path.join("aligned_outputs", args.model, dataset_type, "strategy3", subject_full,
                                  f"{nviews}views")

            for d in (baseline_dir, s1_dir, s2_dir, s3_dir):
                os.makedirs(d, exist_ok=True)

            if not args.no_rerun:
                eye_up = DATASETS[dataset_type].get("eye_up", RERUN_EYE_UP)
                print(f"[RERUN] Using eye_up={eye_up} for dataset={dataset_type}")
                configure_rerun_view_defaults(view_root, eye_up)

            frame_paths = []

            if not args.pgo:
                print(f"\n[STAGE] Baseline: subject={code} views={nviews}")
                target_views = _target_views_for_nviews(nviews, dataset_config, subject_full)
                run_tag = view_root
                baseline_run_reconstruction(
                    model=model,
                    model_type=args.model,
                    dataset_root=dataset_root,
                    target_views=target_views,
                    out_dir=baseline_dir,
                    cache_root=cache_root,
                    run_tag=run_tag,
                    skip_rerun_init=True,
                    no_rerun=args.no_rerun,
                    limit_frames=args.limit_frames,
                    dataset_type=dataset_type,
                )

            frame_paths = _sorted_frame_paths(baseline_dir)
            if len(frame_paths) < 2:
                print(f"[WARN] Not enough baseline frames for subject={code} views={nviews}; skipping strategies.")
                continue

            # Log GT sequence only if baseline was skipped.
            if args.pgo and not args.no_rerun:
                try:
                    log_gt_sequence(frame_paths, log_root=view_root)
                except Exception as e:
                    print(f"[RERUN][WARN] log_gt_sequence failed for {code} {nviews}views: {e}")

            if not args.pgo:
                print(f"\n[STAGE] Strategy 1: subject={code} views={nviews}")
                s1_start = time.perf_counter()
                tf_s1 = strategy1_reference(frame_paths, dataset_root, dataset_type=dataset_type)
                s_g1, R_g1, tr_g1 = solve_final_gt_registration(frame_paths, tf_s1, dataset_root, use_static_mask=False,
                                                                dataset_type=dataset_type)
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
                    dataset_type=dataset_type,
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
                        color=[255, 0, 0],
                        dataset_root=dataset_root,
                        log_root=view_root,
                        dataset_type=dataset_type,
                    )

                print(f"\n[STAGE] Strategy 2: subject={code} views={nviews}")
                s2_start = time.perf_counter()
                tf_s2 = strategy2_hierarchical(frame_paths, dataset_root, dataset_type=dataset_type)
                s_g2, R_g2, tr_g2 = solve_final_gt_registration(frame_paths, tf_s2, dataset_root, use_static_mask=False,
                                                                dataset_type=dataset_type)
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
                    dataset_type=dataset_type,
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
                        color=[255, 0, 255],
                        dataset_root=dataset_root,
                        log_root=view_root,
                        dataset_type=dataset_type,
                    )

            if not args.opt:
                print(f"\n[STAGE] Strategy 3 (PGO): subject={code} views={nviews}")
            if args.opt:
                print(f"\n[STAGE] Temporal Optimization (base={args.opt_base}): subject={code} views={nviews}")
                from pi3.utils.temporal_optimizer import ensure_base_strategy_exists, optimize_temporal_consistency

                base_in_dir = ensure_base_strategy_exists(
                    subject_full, nviews,
                    dataset_type=dataset_type,
                    base_strategy=args.opt_base,
                    model_name=args.model,
                )
                if base_in_dir is None:
                    print(f"[ERROR] Could not obtain {args.opt_base} outputs for {subject_full}. Cannot run --opt.")
                    continue

                opt_dir = os.path.join("aligned_outputs", args.model, dataset_type, "opt", subject_full,
                                       f"{nviews}views")
                os.makedirs(opt_dir, exist_ok=True)

                base_frame_paths = _sorted_frame_paths(base_in_dir)
                if len(base_frame_paths) < 2:
                    print(f"[ERROR] Not enough frames in {base_in_dir}.")
                    continue

                optimize_temporal_consistency(
                    frame_paths=base_frame_paths,
                    out_dir=opt_dir,
                    dataset_root=dataset_root,
                    sigma=args.opt_sigma,
                    alpha=args.opt_alpha,
                    dataset_type=args.data,
                )
            elif not args.opt:
                s3_start = time.perf_counter()
                tf_s3 = strategy3_pgo(frame_paths, dataset_root, num_iters=50, dataset_type=dataset_type)
                s_g3, R_g3, tr_g3 = solve_final_gt_registration(frame_paths, tf_s3, dataset_root, use_static_mask=False,
                                                                dataset_type=dataset_type)
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
                    dataset_type=dataset_type,
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
                        color=[0, 255, 255],
                        dataset_root=dataset_root,
                        log_root=view_root,
                        dataset_type=dataset_type,
                    )

        print(f"\n[INFO] Evaluating subject {code} across methods/views ...")
        if not args.pgo:
            _run_eval(code, view_counts, dataset_type, args.model, opt=args.opt)

    # Aggregate results across selected subjects only.
    csv_files = []
    for code in codes:
        safe_code = code.replace("/", "_")
        csv_path = f"eval_summary_{args.model}_{dataset_type}_{safe_code}.csv"
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
    print(f"CROSS-SUBJECT AGGREGATED RESULTS — {dataset_type.upper()} (MEAN ACROSS PROCESSED SUBJECTS)")
    print("=" * 80)

    pd.set_option("display.precision", 5)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", None)

    # Enforce logical column ordering
    cols_to_show = [
        'strategy', 'n_frames', 'chamfer', 'delta_consistency', 'completeness',
        'static_comp', 'dyn_comp', 'static_acc', 'dyn_acc', 'motion_gap',
        'align_frames', 'ate', 'rpe', 'rot_error', 'focal_error', 'pp_error',
        'jitter_mean', 'jitter_std', 'jitter_p95', 'jitter_max', 'drift_mean', 'hf_jitter'
    ]
    # Filter to only include columns that actually exist in the aggregated DataFrame
    cols_to_show = [c for c in cols_to_show if c in aggregated.columns]
    # Add any remaining columns at the end
    remaining = [c for c in aggregated.columns if c not in cols_to_show]
    cols_to_show += remaining

    print(aggregated[cols_to_show].to_string(index=False))

    out_file = f"eval_summary_{args.model}_{dataset_type}_ALL_SUBJECTS.csv"
    aggregated.to_csv(out_file, index=False)
    print(f"\n[INFO] Aggregated results saved to {out_file}")


if __name__ == "__main__":
    main()
