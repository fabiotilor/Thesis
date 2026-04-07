#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from mast3r.utils.temporal_metrics import (
    compute_chamfer_distance,
    compute_accuracy,
    compute_completeness,
    split_points_by_mask
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Custom directory containing frame_*.npz files")
    args = parser.parse_args()

    np.random.seed(42)  # For reproducibility

    # Determine base directory prefix
    if args.input_dir:
        in_dir = args.input_dir
        # Derived out_plot_dir from input_dir
        out_plot_dir = in_dir.replace("aligned_outputs", "plots")
    else:
        in_dir = "aligned_outputs"
        out_plot_dir = "plots"

    files = sorted(glob.glob(f"{in_dir}/frame_*.npz"))
    if not files:
        print(f"No aligned output files found in {in_dir}. Please run align_reconstruction_umeyama.py first.")
        return

    chamfer_distances = []
    completeness_scores = []
    static_accuracies = []
    dynamic_accuracies = []

    taus = [0.005, 0.01, 0.02, 0.03, 0.05]
    static_accuracies_taus = {tau: [] for tau in taus}
    dynamic_accuracies_taus = {tau: [] for tau in taus}

    point_sequence = []

    print(f"Loading {len(files)} frames for temporal evaluation from {in_dir}...")

    # Process frames
    for i, f in enumerate(files):
        data = np.load(f)
        gt_pts = data['gt_pts']
        est_pts = data['aligned_pts']

        # Compute single frame metrics using ALL points
        chamfer = compute_chamfer_distance(est_pts, gt_pts)
        completeness = compute_completeness(est_pts, gt_pts, tau=0.01)

        chamfer_distances.append(chamfer)
        completeness_scores.append(completeness)

        if 'masks_2d' in data and 'Ks' in data and 'R_ts' in data:
            masks_2d = data['masks_2d']
            Ks = data['Ks']
            R_ts = data['R_ts']

            static_pts, dynamic_pts = split_points_by_mask(est_pts, masks_2d, Ks, R_ts)

            for tau in taus:
                s_acc = compute_accuracy(static_pts, gt_pts, tau=tau) if len(static_pts) > 0 else np.nan
                d_acc = compute_accuracy(dynamic_pts, gt_pts, tau=tau) if len(dynamic_pts) > 0 else np.nan
                static_accuracies_taus[tau].append(s_acc)
                dynamic_accuracies_taus[tau].append(d_acc)

            static_accuracies.append(static_accuracies_taus[0.01][-1])
            dynamic_accuracies.append(dynamic_accuracies_taus[0.01][-1])
        else:
            for tau in taus:
                static_accuracies_taus[tau].append(np.nan)
                dynamic_accuracies_taus[tau].append(np.nan)
            static_accuracies.append(np.nan)
            dynamic_accuracies.append(np.nan)

        # For temporal trajectory metrics, establish point correspondence across time.
        # We establish identity by nearest neighbor matching to the first frame's points (no subsampling).
        if i == 0:
            point_sequence.append(est_pts)
            base_points = est_pts
        else:
            tree = cKDTree(est_pts)
            _, idx = tree.query(base_points, k=1)
            matched_est = est_pts[idx]
            point_sequence.append(matched_est)

    point_sequence = np.stack(point_sequence, axis=0)  # Shape: (T, N, 3)

    valid_static = np.array(static_accuracies)[~np.isnan(static_accuracies)]
    valid_dynamic = np.array(dynamic_accuracies)[~np.isnan(dynamic_accuracies)]

    mean_static = np.mean(valid_static) if len(valid_static) > 0 else np.nan
    mean_dynamic = np.mean(valid_dynamic) if len(valid_dynamic) > 0 else np.nan
    motion_gap = mean_static - mean_dynamic

    print("\n--- Metrics Summary ---")
    print(f"Mean Chamfer Distance:   {np.mean(chamfer_distances):.5f}")
    print(f"Mean Completeness:       {np.nanmean(completeness_scores):.5f}")
    print(f"Mean Static Accuracy:    {mean_static:.5f}")
    print(f"Mean Dynamic Accuracy:   {mean_dynamic:.5f}")
    print(f"Motion Gap (Static-Dyn): {motion_gap:.5f}")

    # Generate Plots
    os.makedirs(out_plot_dir, exist_ok=True)
    frames = np.arange(len(files))

    # Plot 2: Chamfer Distance
    plt.figure()
    plt.plot(frames, chamfer_distances, marker='s', color='green')
    plt.title('Frame vs Chamfer Distance')
    plt.xlabel('Frame')
    plt.ylabel('Chamfer Distance')
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, 'chamfer_distance.png'))
    plt.close()

    # Plot 5: Static vs Dynamic Accuracy
    plt.figure()
    plt.plot(frames, static_accuracies, marker='o', color='blue', label='Static Accuracy')
    plt.plot(frames, dynamic_accuracies, marker='x', color='red', label='Dynamic Accuracy')
    plt.title('Frame vs Split Accuracy')
    plt.xlabel('Frame')
    plt.ylabel('Accuracy (% < 1cm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, 'split_accuracy.png'))
    plt.close()

    # Plot 6: Accuracy-Threshold Curve
    mean_static_taus = [np.nanmean(static_accuracies_taus[t]) for t in taus]
    mean_dynamic_taus = [np.nanmean(dynamic_accuracies_taus[t]) for t in taus]

    plt.figure()
    plt.plot(taus, [m * 100 for m in mean_static_taus], marker='o', color='blue', label='Static')
    plt.plot(taus, [m * 100 for m in mean_dynamic_taus], marker='x', color='red', label='Dynamic')
    plt.title('Accuracy-Threshold Curve')
    plt.xlabel('Distance Threshold (m)')
    plt.ylabel('Mean Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, 'accuracy_threshold_curve.png'))
    plt.close()

    print(f"\nPlots saved in {out_plot_dir}/ directory.")


if __name__ == '__main__':
    main()
