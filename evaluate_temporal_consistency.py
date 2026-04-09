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


def evaluate_configuration(in_dir, out_plot_dir, plot_prefix=""):
    files = sorted(glob.glob(os.path.join(in_dir, "frame_*.npz")))
    if not files:
        print(f"[WARN] No aligned output files found in {in_dir}; skipping.")
        return None

    chamfer_distances = []
    completeness_scores = []
    static_accuracies = []
    dynamic_accuracies = []

    taus = [0.01]#0.005, 0.01, 0.02, 0.03, 0.05
    static_accuracies_taus = {tau: [] for tau in taus}
    dynamic_accuracies_taus = {tau: [] for tau in taus}

    point_sequence = []
    print(f"Loading {len(files)} frames for temporal evaluation from {in_dir}...")

    for i, f in enumerate(files):
        data = np.load(f)
        gt_pts = data['gt_pts']
        est_pts = data['aligned_pts']

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

    os.makedirs(out_plot_dir, exist_ok=True)
    frames = np.arange(len(files))

    plt.figure()
    plt.plot(frames, chamfer_distances, marker='s', color='green')
    plt.title('Frame vs Chamfer Distance')
    plt.xlabel('Frame')
    plt.ylabel('Chamfer Distance')
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, f'{plot_prefix}chamfer_distance.png'))
    plt.close()

    plt.figure()
    plt.plot(frames, static_accuracies, marker='o', color='blue', label='Static Accuracy')
    plt.plot(frames, dynamic_accuracies, marker='x', color='red', label='Dynamic Accuracy')
    plt.title('Frame vs Split Accuracy')
    plt.xlabel('Frame')
    plt.ylabel('Accuracy (% < 1cm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, f'{plot_prefix}split_accuracy.png'))
    plt.close()

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
    plt.savefig(os.path.join(out_plot_dir, f'{plot_prefix}accuracy_threshold_curve.png'))
    plt.close()

    return {
        "mean_static_accuracy": mean_static,
        "mean_dynamic_accuracy": mean_dynamic,
        "static_accuracies": static_accuracies,
        "dynamic_accuracies": dynamic_accuracies,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Custom directory containing frame_*.npz files")
    args = parser.parse_args()

    np.random.seed(42)  # For reproducibility

    results = {}
    os.makedirs("plots", exist_ok=True)

    if args.input_dir:
        in_dir = args.input_dir
        out_plot_dir = in_dir.replace("aligned_outputs", "plots")
        metrics = evaluate_configuration(in_dir, out_plot_dir)
        if metrics is None:
            print(f"No aligned output files found in {in_dir}. Please run align_reconstruction_umeyama.py first.")
            return
        results["custom"] = {
            "static": metrics["mean_static_accuracy"],
            "dynamic": metrics["mean_dynamic_accuracy"],
        }
        print(f"\nPlots saved in {out_plot_dir}/ directory.")
        return

    base_in_dir = "aligned_outputs"
    out_plot_dir = "plots"
    camera_counts = [2, 3, 4]
    available_counts = []

    for cam_count in camera_counts:
        in_dir = os.path.join(base_in_dir, f"{cam_count}views")
        if not os.path.isdir(in_dir):
            print(f"[INFO] Missing directory: {in_dir}, skipping.")
            continue

        metrics = evaluate_configuration(
            in_dir=in_dir,
            out_plot_dir=out_plot_dir,
            plot_prefix=f"{cam_count}views_",
        )
        if metrics is None:
            continue

        available_counts.append(cam_count)
        results[cam_count] = {
            "static": metrics["mean_static_accuracy"],
            "dynamic": metrics["mean_dynamic_accuracy"],
        }

    if not available_counts:
        fallback_metrics = evaluate_configuration(base_in_dir, out_plot_dir)
        if fallback_metrics is None:
            print("No valid aligned outputs found. Please run align_reconstruction_umeyama.py first.")
            return
        print(f"\nPlots saved in {out_plot_dir}/ directory.")
        return

    static_means = [results[c]["static"] for c in available_counts]
    dynamic_means = [results[c]["dynamic"] for c in available_counts]

    plt.figure()
    plt.plot(available_counts, static_means, marker='o', color='blue', label='Static Accuracy')
    plt.plot(available_counts, dynamic_means, marker='x', color='red', label='Dynamic Accuracy')
    plt.title('Static vs Dynamic Accuracy vs Number of Cameras')
    plt.xlabel('Number of Cameras')
    plt.ylabel('Accuracy (% < 1cm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, 'accuracy_vs_cameras.png'))
    plt.close()

    print("\nPer-configuration results:")
    for cam_count in available_counts:
        print(
            f"  {cam_count} views -> "
            f"static={results[cam_count]['static']:.5f}, "
            f"dynamic={results[cam_count]['dynamic']:.5f}"
        )
    print(f"\nPlots saved in {out_plot_dir}/ directory.")


if __name__ == '__main__':
    main()
