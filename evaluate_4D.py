#!/usr/bin/env python3
import os
import csv
import glob
import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vggt.utils.umeyama_alignment import apply_similarity_transform
from vggt.utils.temporal_metrics import (
    compute_chamfer_distance,
    compute_accuracy,
    compute_completeness,
    split_points_by_mask,
    compute_static_jitter,
    compute_camera_metrics
)
from vggt.utils.alignment_4d import normalize_spatial_dims, normalize_array
from eval_config import SUBJECT_NAMES, SUBJECT_BY_CODE


def print_metrics_summary(results_df, label):
    """Prints a comparison table for all strategies."""
    print(f"\n=== Performance Summary: {label} ===")
    pd.set_option('display.precision', 5)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_columns', None)

    cols_to_show = [
        'strategy', 'n_frames',
        'align_frames',
        'chamfer', 'delta_consistency', 'completeness', 'static_comp', 'dyn_comp', 'static_acc', 'dyn_acc', 'motion_gap',
        'ate', 'rpe', 'rot_error', 'focal_error', 'pp_error',
        'jitter_mean', 'jitter_std', 'jitter_p95', 'jitter_max',
        'drift_mean', 'hf_jitter'
    ]
    # Filter only columns that exist
    cols_to_show = [c for c in cols_to_show if c in results_df.columns]
    print(results_df[cols_to_show].to_string(index=False))
    print("=" * (len(label) + 25))


def evaluate_strategy_dir(in_dir, out_plot_dir, strategy_label=""):
    files = sorted(glob.glob(os.path.join(in_dir, "frame_*.npz")))
    if not files:
        return None

    print(f"  [EVAL] {strategy_label}: {len(files)} frames...")

    cham_dist, comp_score, s_acc_list, d_acc_list = [], [], [], []
    s_comp_list, d_comp_list = [], []
    all_est_poses, all_est_intrinsics = [], []
    all_gt_poses, all_gt_intrinsics = [], []

    # For Jitter
    # For Jitter & Camera Tracking
    all_pointmaps_mv = []
    all_masks_mv = []
    ate_list, rpe_list, rot_err_list, focal_err_list, pp_err_list = [], [], [], [], []

    for f in files:
        data = np.load(f)
        gt_pts, est_pts = data['gt_pts'], data['aligned_pts']

        # Squeeze out nans if any (sanity check)
        valid_est = ~np.any(np.isnan(est_pts), axis=-1)
        est_pts = est_pts[valid_est]

        ks, rts = data['Ks'], data['R_ts']
        m_2d = data['masks_2d']

        if len(est_pts) > 0 and len(gt_pts) > 0:
            cham_dist.append(compute_chamfer_distance(est_pts, gt_pts))
            comp_score.append(compute_completeness(est_pts, gt_pts, tau=0.01))

            s_p, d_p = split_points_by_mask(est_pts, m_2d, ks, rts)
            g_s, g_d = split_points_by_mask(gt_pts, m_2d, ks, rts)

            s_acc_list.append(compute_accuracy(s_p, g_s, tau=0.01) if len(s_p) > 0 else np.nan)
            d_acc_list.append(compute_accuracy(d_p, g_d, tau=0.01) if len(d_p) > 0 else np.nan)

            s_comp_list.append(compute_completeness(s_p, g_s, tau=0.01) if len(g_s) > 0 else np.nan)
            d_comp_list.append(compute_completeness(d_p, g_d, tau=0.01) if len(g_d) > 0 else np.nan)
        else:
            cham_dist.append(np.nan)
            comp_score.append(np.nan)
            s_acc_list.append(np.nan)
            d_acc_list.append(np.nan)
            s_comp_list.append(np.nan)
            d_comp_list.append(np.nan)

        s_val, R_val, tr_val = data['scale'], data['R'], data['tr']

        if 'est_poses' in data and data['est_poses'] is not None and data['est_poses'].ndim >= 3:
            e_p = data['est_poses']
            g_p = np.array([np.linalg.inv(rt) for rt in rts])
            e_i = data['est_intrinsics']
            cam_mets = compute_camera_metrics(e_p, g_p, e_i, ks, s_val, R_val, tr_val)
            if not np.isnan(cam_mets['ate']):
                ate_list.append(cam_mets['ate'])
                rpe_list.append(cam_mets['rpe'])
                rot_err_list.append(cam_mets['rot_error'])
                focal_err_list.append(cam_mets['focal_error'])
                pp_err_list.append(cam_mets['pp_error'])

        pointmap_key = 'world_points' if 'world_points' in data else ('pointmaps' if 'pointmaps' in data else None)
        if pointmap_key is not None:
            V, H, W = normalize_spatial_dims(data)
            pm = normalize_array(data[pointmap_key], V, H, W)
            m_norm = normalize_array(m_2d, V, H, W, is_mask=True)
            aligned_pm = np.empty_like(pm)
            for vi in range(V):
                aligned_pm[vi] = apply_similarity_transform(
                    pm[vi].reshape(-1, 3), s_val, R_val, tr_val
                ).reshape(H, W, 3)
            all_pointmaps_mv.append(aligned_pm)
            all_masks_mv.append(m_norm.astype(bool))

    # Calculate Aggregated Metrics
    m_static = np.nanmean(s_acc_list)
    m_dyn = np.nanmean(d_acc_list)

    metrics = {
        'strategy': strategy_label,
        'n_frames': len(files),
        'chamfer': np.nanmean(cham_dist),
        'completeness': np.nanmean(comp_score),
        'static_comp': np.nanmean(s_comp_list),
        'dyn_comp': np.nanmean(d_comp_list),
        'static_acc': m_static,
        'dyn_acc': m_dyn,
        'motion_gap': m_static - m_dyn if not np.isnan(m_static) and not np.isnan(m_dyn) else np.nan
    }

    timing_path = os.path.join(in_dir, "timing.json")
    if os.path.exists(timing_path):
        try:
            with open(timing_path, "r", encoding="utf-8") as f:
                timing = json.load(f)
            metrics["align_frames"] = int(timing.get("n_frames", len(files)))
        except Exception as e:
            print(f"  [WARN] Failed to read timing file {timing_path}: {e}")

    if ate_list:
        cam_metrics = {
            'ate': float(np.nanmean(ate_list)),
            'rpe': float(np.nanmean(rpe_list)),
            'rot_error': float(np.nanmean(rot_err_list)),
            'focal_error': float(np.nanmean(focal_err_list)),
            'pp_error': float(np.nanmean(pp_err_list)),
        }
        metrics.update(cam_metrics)

    jitter = None
    if len(all_pointmaps_mv) >= 2:
        # Optimization: compute_static_jitter uses the pre-computed masks_2d from the NPZ
        # No Farneback flow is recalculated here.
        jitter = compute_static_jitter(all_pointmaps_mv, all_masks_mv, n_anchors=5000)
        if jitter: metrics.update(jitter)

    # Simple temporal plots
    os.makedirs(out_plot_dir, exist_ok=True)
    frames = np.arange(len(files))
    plt.figure()
    plt.plot(frames, cham_dist, 'g-s', label='Chamfer')
    plt.title(f'Chamfer Distance - {strategy_label}')
    plt.savefig(os.path.join(out_plot_dir, f'chamfer_{strategy_label}.png'))
    plt.close()

    return metrics


def add_delta_consistency(results_df):
    """
    Δconsistency = Chamfer4D - Chamfer3D (baseline), matched per view-count.
    Supports both:
      - new labels: baseline_2views, strategy1_2views, strategy2_2views, strategy3_2views
      - legacy labels: 2views, Strategy_1_2views, Strategy_2_2views, Strategy_3_2views
    """
    if results_df.empty or "strategy" not in results_df.columns or "chamfer" not in results_df.columns:
        return results_df

    df = results_df.copy()
    df["delta_consistency"] = np.nan

    def extract_view_suffix(strategy_label):
        match = re.search(r"(\d+views)$", str(strategy_label))
        return match.group(1) if match else None

    def is_baseline_label(strategy_label):
        label = str(strategy_label)
        return bool(re.fullmatch(r"\d+views", label) or label.startswith("baseline_"))

    baseline_by_view = {}
    for _, row in df.iterrows():
        strategy_label = row["strategy"]
        if is_baseline_label(strategy_label):
            view_suffix = extract_view_suffix(strategy_label)
            if view_suffix is not None:
                baseline_by_view[view_suffix] = row["chamfer"]

    for idx, row in df.iterrows():
        strategy_label = row["strategy"]
        if is_baseline_label(strategy_label):
            continue
        view_suffix = extract_view_suffix(strategy_label)
        if view_suffix is None:
            continue
        baseline_chamfer = baseline_by_view.get(view_suffix)
        if baseline_chamfer is None or np.isnan(baseline_chamfer):
            continue
        df.at[idx, "delta_consistency"] = row["chamfer"] - baseline_chamfer

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pgo", action="store_true", help="Evaluate only Strategy 3 outputs.")
    parser.add_argument("--views", nargs="+", type=int, help="Optional view counts to evaluate (e.g. --views 2 3 4).")
    for code in SUBJECT_BY_CODE.keys(): parser.add_argument(f"--{code}", action="store_true")
    args = parser.parse_args()

    selected = [k for k in SUBJECT_BY_CODE.keys() if getattr(args, k)]
    subjects = selected if not args.all else [s.split("-subject-")[1][:2] for s in SUBJECT_NAMES]
    if not subjects: subjects = ["01"]  # Default

    method_roots = ["baseline", "strategy1", "strategy2", "strategy3"]
    if args.pgo:
        method_roots = ["strategy3"]

    view_set = set(args.views) if args.views else None

    def _is_view_dir(name: str) -> bool:
        return re.match(r"^\d+views$", name) is not None

    for scode in subjects:
        subject_full = SUBJECT_BY_CODE.get(scode)
        if not subject_full: continue

        subject_results = []

        # New layout: aligned_outputs/{method}/{subject_full}/{Nviews}/
        any_new_found = False
        for method in method_roots:
            subject_dir = os.path.join("aligned_outputs", method, subject_full)
            if not os.path.exists(subject_dir):
                print(f"[WARN] No outputs for method={method} subject={subject_full}")
                continue

            view_dirs = sorted(
                [d for d in os.listdir(subject_dir) if _is_view_dir(d) and os.path.isdir(os.path.join(subject_dir, d))]
            )
            if view_set is not None:
                view_dirs = [d for d in view_dirs if int(d.split("views")[0]) in view_set]

            if not view_dirs:
                print(f"[WARN] No view folders found for method={method} subject={subject_full}")
                continue

            any_new_found = True
            for view_dir in view_dirs:
                in_dir = os.path.join(subject_dir, view_dir)
                plot_dir = os.path.join("plots", subject_full, method, view_dir)
                strategy_label = f"{method}_{view_dir}"
                res = evaluate_strategy_dir(in_dir, plot_dir, strategy_label=strategy_label)
                if res:
                    res["subject"] = subject_full
                    subject_results.append(res)

        if subject_results:
            df = add_delta_consistency(pd.DataFrame(subject_results))
            print_metrics_summary(df, subject_full)
            out_csv = f"eval_summary_{scode}.csv"
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Saved combined report to {out_csv}")
            continue

        # Legacy fallback: aligned_outputs/<subject_full>/<Strategy_*/2views/...>
        base_dir = os.path.join("aligned_outputs", subject_full)
        if not os.path.exists(base_dir):
            if not any_new_found:
                print(f"[WARN] No aligned outputs found for {subject_full}")
            continue

        strategies = sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        )
        if args.views:
            wanted_suffixes = tuple(f"_{v}views" for v in args.views)
            strategies = [s for s in strategies if s.endswith(wanted_suffixes)]
        if args.pgo:
            strategies = [s for s in strategies if s.startswith("Strategy_3_")]

        if not strategies:
            print(f"[WARN] No strategies found in {base_dir}")
            continue

        subject_results = []
        plot_root = os.path.join("plots", subject_full)
        for strat in strategies:
            in_dir = os.path.join(base_dir, strat)
            plot_dir = os.path.join(plot_root, strat)
            res = evaluate_strategy_dir(in_dir, plot_dir, strategy_label=strat)
            if res:
                res["subject"] = subject_full
                subject_results.append(res)

        if subject_results:
            df = add_delta_consistency(pd.DataFrame(subject_results))
            print_metrics_summary(df, subject_full)
            out_csv = f"eval_summary_{scode}.csv"
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Saved combined report to {out_csv}")


if __name__ == "__main__":
    main()
