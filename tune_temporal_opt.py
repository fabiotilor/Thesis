#!/usr/bin/env python3
"""
Hyperparameter sweep for the temporal consistency optimizer.

Sweeps over (sigma, alpha) pairs for the confidence-weighted Gaussian
temporal filter, evaluates jitter / chamfer / accuracy for each, and
generates a Pareto-frontier plot.

Usage:
    python tune_temporal_opt.py --subjects 01 --views 4
    python tune_temporal_opt.py --subjects 01 --views 4 --base strategy1
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

from vggt.utils.temporal_optimizer import (
    load_and_align_frames,
    confidence_weighted_temporal_smooth,
    ensure_base_strategy_exists,
)
from vggt.utils.temporal_metrics import (
    compute_chamfer_distance,
    compute_accuracy,
    compute_static_jitter,
)
from vggt.utils.alignment_4d import normalize_spatial_dims
from vggt.utils.camera_utils import discover_view_name
from vggt.utils.gt import build_gt_validity_masks
from eval_config import DATASETS, CONF_PERCENTILE


def run_sweep(subject_name, dataset_type, views, base_strategy="strategy2"):
    # ── 1. Ensure base-strategy outputs exist (auto-compute if missing) ──
    base_in_dir = ensure_base_strategy_exists(
        subject_name, views,
        dataset_type=dataset_type,
        base_strategy=base_strategy,
    )
    if base_in_dir is None:
        print(f"[ERROR] Could not obtain {base_strategy} outputs for "
              f"{subject_name} {views}views – aborting sweep.")
        return

    frame_paths = sorted(
        glob.glob(os.path.join(base_in_dir, "frame_*.npz")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]),
    )
    if len(frame_paths) < 2:
        print(f"[ERROR] Not enough frames in {base_in_dir}")
        return

    print(f"Loading {len(frame_paths)} frames from {base_in_dir}...")
    dataset_root = os.path.join(DATASETS[dataset_type]["root"], subject_name)
    all_pmaps, all_confs, all_masks, all_data = load_and_align_frames(frame_paths)

    if all_pmaps is None:
        print("Failed to load frames.")
        return


    all_validity_masks = []
    for i, data in enumerate(all_data):
        V, H, W = normalize_spatial_dims(data)
        t = int(data["frame_idx"])
        ks = data["Ks"]
        if 'view_names' in data:
            view_names = (data['view_names'].tolist()
                          if hasattr(data['view_names'], 'tolist')
                          else list(data['view_names']))
        else:
            view_names = [
                discover_view_name(dataset_root, k, dataset_type=dataset_type)
                for k in ks
            ]
        vmasks = build_gt_validity_masks(
            t, view_names, dataset_root,
            target_hw=(H, W), dataset_type=dataset_type,
        )
        all_validity_masks.append(np.array([
            vmask if vmask is not None else np.ones((H, W), dtype=bool)
            for vmask in vmasks
        ], dtype=bool))

    # ── 2. Parameter grid ────────────────────────────────────────────────
    sigmas = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0]
    alphas = [0.2, 0.5, 0.8, 1.0]

    # Include original baseline (no smoothing)
    configs = [(0.0, 0.0)] + [(s, a) for s in sigmas for a in alphas]

    results = []

    print("\nStarting parameter sweep...")
    print(f"{'Sigma':<6} | {'Alpha':<6} | {'Jitter (m)':<12} | {'Chamfer':<10} | {'Static Acc':<10}")
    print("-" * 55)

    for sigma, alpha in configs:
        smoothed_pmaps = confidence_weighted_temporal_smooth(
            all_pmaps, all_confs, all_masks, sigma, alpha,
        )

        jitter_results = compute_static_jitter(
            pointmaps_per_frame=list(smoothed_pmaps),
            masks_per_frame=list(all_masks),
            validity_masks_per_frame=all_validity_masks,
            confidences_per_frame=list(all_confs) if all_confs is not None else None,
            conf_percentile=CONF_PERCENTILE,
            n_anchors=5000,
        )
        jitter_mean = jitter_results.get('jitter_mean', np.nan)

        chamfer_dists = []
        static_accs = []

        for i, data in enumerate(all_data):
            V, H, W = normalize_spatial_dims(data)
            t = int(data["frame_idx"])
            ks = data["Ks"]

            if 'view_names' in data:
                view_names = (data['view_names'].tolist()
                              if hasattr(data['view_names'], 'tolist')
                              else list(data['view_names']))
            else:
                view_names = [discover_view_name(dataset_root, k) for k in ks]

            vmasks = build_gt_validity_masks(
                t, view_names, dataset_root,
                target_hw=(H, W), dataset_type=dataset_type,
            )

            all_pts = []
            conf = all_confs[i]

            for v in range(V):
                mask = np.ones((H, W), dtype=bool)
                if vmasks[v] is not None:
                    mask &= vmasks[v]
                if conf is not None:
                    thr = np.percentile(conf[v], 100 * (1 - CONF_PERCENTILE))
                    mask &= conf[v] > thr

                p_v = smoothed_pmaps[i, v][mask]
                if len(p_v) > 0:
                    all_pts.append(p_v)

            aligned_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
            gt_pts = data["gt_pts"]

            chamfer_dists.append(compute_chamfer_distance(aligned_pts, gt_pts))

            if len(aligned_pts) > 0 and len(gt_pts) > 0:
                acc = compute_accuracy(aligned_pts, gt_pts, tau=0.01)
                static_accs.append(acc)

        mean_chamfer = np.mean(chamfer_dists)
        mean_acc = np.nanmean(static_accs)

        results.append({
            'sigma': sigma,
            'alpha': alpha,
            'jitter': jitter_mean,
            'chamfer': mean_chamfer,
            'accuracy': mean_acc,
        })

        print(f"{sigma:<6.2f} | {alpha:<6.2f} | {jitter_mean:<12.5f} | "
              f"{mean_chamfer:<10.5f} | {mean_acc:<10.5f}")

    # ── 3. Plot Pareto Frontier ──────────────────────────────────────────
    jitters = [r['jitter'] for r in results]
    chamfers = [r['chamfer'] for r in results]
    labels  = [f"s={r['sigma']},a={r['alpha']}" if r['sigma'] > 0
               else "Baseline" for r in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(jitters, chamfers, color='blue')

    for i, label in enumerate(labels):
        if "Baseline" in label:
            plt.scatter([jitters[i]], [chamfers[i]], color='red', s=100, label='Baseline')
            plt.annotate("Baseline", (jitters[i], chamfers[i]),
                         xytext=(5, 5), textcoords='offset points',
                         color='red', fontweight='bold')
        else:
            plt.annotate(label, (jitters[i], chamfers[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.title(f'Temporal Optimization Pareto Frontier – {subject_name} {views}views')
    plt.xlabel('Static Jitter (m) ↓')
    plt.ylabel('Chamfer Distance (m) ↓')
    plt.grid(True, linestyle='--', alpha=0.7)

    out_plot_dir = os.path.join("plots", "tune_opt")
    os.makedirs(out_plot_dir, exist_ok=True)
    plot_path = os.path.join(out_plot_dir,
                             f'pareto_{subject_name}_{views}views.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    csv_path = os.path.join(out_plot_dir,
                            f'sweep_{subject_name}_{views}views.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['sigma', 'alpha', 'jitter', 'chamfer', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[INFO] Saved Pareto plot to {plot_path}")
    print(f"[INFO] Saved sweep results to {csv_path}")


if __name__ == '__main__':
    from vggt.utils.rerun_logging import add_dataset_args, get_selected_subjects
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    parser.add_argument("--views", type=int, default=4,
                        help="Number of views to evaluate.")
    parser.add_argument("--base", type=str, default="strategy2",
                        help="Base alignment strategy to smooth (default: strategy2).")
    args = parser.parse_args()

    dataset_type = args.data
    subjects, codes = get_selected_subjects(args)

    for subject_name in subjects:
        run_sweep(subject_name, dataset_type, args.views, args.base)
