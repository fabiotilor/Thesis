#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from mast3r.utils.temporal_metrics import (
    compute_l2_error,
    compute_chamfer_distance,
    compute_temporal_jitter,
    compute_temporal_variance
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

    l2_errors = []
    chamfer_distances = []
    point_sequence = []

    print(f"Loading {len(files)} frames for temporal evaluation from {in_dir}...")

    # Process frames
    for i, f in enumerate(files):
        data = np.load(f)
        gt_pts = data['gt_pts']
        est_pts = data['aligned_pts']

        # Compute single frame metrics using ALL points
        l2 = compute_l2_error(est_pts, gt_pts)
        chamfer = compute_chamfer_distance(est_pts, gt_pts)

        l2_errors.append(l2)
        chamfer_distances.append(chamfer)

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

    jitter_per_frame = compute_temporal_jitter(point_sequence)
    temporal_variance_scalar = compute_temporal_variance(point_sequence)

    print("\n--- Metrics Summary ---")
    print(f"Mean L2 Error:           {np.mean(l2_errors):.5f}")
    print(f"Mean Chamfer Distance:   {np.mean(chamfer_distances):.5f}")
    print(f"Mean Temporal Jitter:    {np.mean(jitter_per_frame):.5f}")
    print(f"Temporal Variance:       {temporal_variance_scalar:.5f}")

    # Generate Plots
    os.makedirs(out_plot_dir, exist_ok=True)
    frames = np.arange(len(files))

    # Plot 1: L2 Error
    plt.figure()
    plt.plot(frames, l2_errors, marker='o', color='blue')
    plt.title('Frame vs Mean L2 Error')
    plt.xlabel('Frame')
    plt.ylabel('L2 Reconstruction Error')
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, 'l2_error.png'))
    plt.close()

    # Plot 2: Chamfer Distance
    plt.figure()
    plt.plot(frames, chamfer_distances, marker='s', color='green')
    plt.title('Frame vs Chamfer Distance')
    plt.xlabel('Frame')
    plt.ylabel('Chamfer Distance')
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, 'chamfer_distance.png'))
    plt.close()

    # Plot 3: Temporal Jitter
    plt.figure()
    plt.plot(frames[1:], jitter_per_frame, marker='^', color='orange')
    plt.title('Frame vs Temporal Jitter')
    plt.xlabel('Frame')
    plt.ylabel('Temporal Jitter')
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, 'temporal_jitter.png'))
    plt.close()

    # Plot 4: Temporal Variance Histogram
    variance = np.var(point_sequence, axis=0)
    norm_variance = np.linalg.norm(variance, axis=-1)

    plt.figure()
    plt.hist(norm_variance, bins=30, color='purple', alpha=0.7)
    plt.title('Histogram of Point Variance')
    plt.xlabel('Temporal Variance ||Var(x,y,z)||')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, 'temporal_variance_histogram.png'))
    plt.close()

    print(f"\nPlots saved in {out_plot_dir}/ directory.")


if __name__ == '__main__':
    main()
