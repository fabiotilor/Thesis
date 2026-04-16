#!/usr/bin/env python3
import os
import csv
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path setup for MASt3R/DUSt3R
import mast3r.utils.path_to_dust3r  # noqa

from mast3r.utils.umeyama_alignment import apply_similarity_transform
from mast3r.utils.temporal_metrics import (
    compute_chamfer_distance,
    compute_accuracy,
    compute_completeness,
    split_points_by_mask,
    compute_static_jitter,
    compute_camera_metrics
)
from mast3r.utils.alignment_4d import normalize_spatial_dims, normalize_array
from eval_config import SUBJECT_NAMES, SUBJECT_BY_CODE


def print_metrics_summary(results_df, label):
    """Prints a comparison table for all strategies."""
    print(f"\n=== Performance Summary: {label} ===")
    pd.set_option('display.precision', 5)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_columns', None)

    cols_to_show = [
        'strategy', 'n_frames',
        'chamfer', 'completeness', 'static_acc', 'dyn_acc', 'motion_gap',
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
        else:
            cham_dist.append(np.nan)
            comp_score.append(np.nan)
            s_acc_list.append(np.nan)
            d_acc_list.append(np.nan)

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

        if 'pointmaps' in data:
            V, H, W = normalize_spatial_dims(data)
            pm = normalize_array(data['pointmaps'], V, H, W)
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
        'static_acc': m_static,
        'dyn_acc': m_dyn,
        'motion_gap': m_static - m_dyn if not np.isnan(m_static) and not np.isnan(m_dyn) else np.nan
    }

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

    for scode in subjects:
        subject_full = SUBJECT_BY_CODE.get(scode)
        if not subject_full: continue

        base_dir = os.path.join("aligned_outputs", subject_full)
        if not os.path.exists(base_dir):
            print(f"[WARN] No aligned outputs found for {subject_full}")
            continue

        # Discover all strategies (S1, S2, 3D, etc.)
        strategies = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
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
                res['subject'] = subject_full
                subject_results.append(res)

        if subject_results:
            df = pd.DataFrame(subject_results)
            print_metrics_summary(df, subject_full)
            out_csv = f"eval_summary_{scode}.csv"
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Saved combined report to {out_csv}")


if __name__ == "__main__":
    main()
