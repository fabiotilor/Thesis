#!/usr/bin/env python3
import os
import csv
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from vggt.utils.temporal_metrics import (
    compute_chamfer_distance,
    compute_accuracy,
    compute_completeness,
    split_points_by_mask,
    compute_static_jitter,
    compute_camera_metrics
)
from vggt.utils.umeyama_alignment import apply_similarity_transform

SUBJECT_NAMES = [
    "20200709-subject-01__20200709_141754",
    "20200813-subject-02__20200813_145653",
    "20200820-subject-03__20200820_135841",
    "20200903-subject-04__20200903_104428",
    "20200908-subject-05__20200908_144409",
    "20200918-subject-06__20200918_114117",
    "20200928-subject-07__20200928_144906",
    "20201002-subject-08__20201002_110227",
    "20201015-subject-09__20201015_144721",
    "20201022-subject-10__20201022_112651",
]
SUBJECT_BY_CODE = {name.split("subject-")[1][:2]: name for name in SUBJECT_NAMES}


def _plot_scatter_with_regression(x, y, xlabel, ylabel, title, save_path):
    """Plot a scatter with linear regression line and Pearson r annotation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x_v, y_v = x[valid], y[valid]
    if len(x_v) < 3:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_v, y_v, s=30, alpha=0.7)

    # Linear regression
    coeffs = np.polyfit(x_v, y_v, 1)
    x_fit = np.linspace(x_v.min(), x_v.max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r--', linewidth=1.5)

    # Pearson r
    r = np.corrcoef(x_v, y_v)[0, 1]
    ax.annotate(f"r = {r:.3f}", xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def evaluate_configuration(in_dir, out_plot_dir, plot_prefix=""):
    files = sorted(glob.glob(os.path.join(in_dir, "frame_*.npz")))
    if not files:
        print(f"[WARN] No aligned output files found in {in_dir}; skipping.")
        return None

    chamfer_distances = []
    completeness_scores = []
    static_accuracies = []
    dynamic_accuracies = []

    taus = [0.01]  # 0.005, 0.01, 0.02, 0.03, 0.05
    static_accuracies_taus = {tau: [] for tau in taus}
    dynamic_accuracies_taus = {tau: [] for tau in taus}

    point_sequence = []

    ate_list = []
    rpe_list = []
    rot_err_list = []
    focal_err_list = []
    pp_err_list = []

    # Multi-view data for jitter
    all_pointmaps_mv = []  # list of (V, H, W, 3)
    all_masks_mv = []  # list of (V, H, W)
    all_Ks_mv = []  # list of (V, 3, 3)
    all_R_ts_mv = []  # list of (V, 4, 4)

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

            if 'est_poses' in data and 'est_intrinsics' in data and 'scale' in data and 'R' in data and 'tr' in data:
                est_poses = data['est_poses']
                est_intrinsics = data['est_intrinsics']
                gt_poses = np.array([np.linalg.inv(rt) for rt in R_ts])
                s_val = data['scale']
                R_val = data['R']
                tr_val = data['tr']
                cam_mets = compute_camera_metrics(est_poses, gt_poses, est_intrinsics, Ks, s_val, R_val, tr_val)
                if not np.isnan(cam_mets['ate']):
                    ate_list.append(cam_mets['ate'])
                    rpe_list.append(cam_mets['rpe'])
                    rot_err_list.append(cam_mets['rot_error'])
                    focal_err_list.append(cam_mets['focal_error'])
                    pp_err_list.append(cam_mets['pp_error'])

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

        # Collect multi-view data for temporal jitter
        if 'pointmaps' in data and 'masks_2d' in data:
            pmaps = data['pointmaps']  # (V, H, W, 3) or (V, N, 3) flattened
            masks_2d_raw = data['masks_2d']  # (V, H', W')

            # Ensure pointmaps have spatial dims: if (V, N, 3), reshape
            if pmaps.ndim == 3:
                # Flattened: (V, N, 3) — infer H, W from masks
                H_mask, W_mask = masks_2d_raw.shape[1], masks_2d_raw.shape[2]
                N = pmaps.shape[1]
                H_pm = int(np.round(np.sqrt(N * H_mask / float(W_mask))))
                W_pm = N // H_pm
                pmaps = pmaps.reshape(pmaps.shape[0], H_pm, W_pm, 3)

            # Apply Umeyama alignment to each view's pointmap
            if 'R' in data and 'tr' in data:
                s_val, R_val, tr_val = data['scale'], data['R'], data['tr']
                V_views = pmaps.shape[0]
                aligned_pmaps = np.empty_like(pmaps)
                for vi in range(V_views):
                    aligned_pmaps[vi] = apply_similarity_transform(
                        pmaps[vi].reshape(-1, 3), s_val, R_val, tr_val
                    ).reshape(pmaps[vi].shape)
                all_pointmaps_mv.append(aligned_pmaps)
            else:
                all_pointmaps_mv.append(pmaps)

            all_masks_mv.append(masks_2d_raw.astype(bool))

            if 'Ks' in data:
                all_Ks_mv.append(data['Ks'])
            if 'R_ts' in data:
                all_R_ts_mv.append(data['R_ts'])

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
    if ate_list:
        print(f"Mean ATE:                {np.nanmean(ate_list):.5f} m")
        print(f"Mean RPE:                {np.nanmean(rpe_list):.5f} m")
        print(f"Mean Rotation Error:     {np.nanmean(rot_err_list):.4f} deg")
        print(f"Mean Focal Error:        {np.nanmean(focal_err_list):.5f} (rel)")
        print(f"Mean PP Error:           {np.nanmean(pp_err_list):.3f} px")

    # Compute jitter with multi-view fusion
    if len(all_pointmaps_mv) >= 2:
        jitter_results = compute_static_jitter(
            pointmaps_per_frame=all_pointmaps_mv,
            masks_per_frame=all_masks_mv,
            Ks_per_frame=all_Ks_mv if all_Ks_mv else None,
            R_ts_per_frame=all_R_ts_mv if all_R_ts_mv else None,
            n_anchors=5000,
        )
    else:
        jitter_results = {
            'jitter_mean': np.nan, 'jitter_std': np.nan, 'jitter_p95': np.nan,
            'jitter_max': np.nan, 'drift_mean': np.nan, 'hf_jitter': np.nan,
            'per_frame_jitter': np.array([]), 'n_anchors': 0, 'n_frames': 0,
        }

    jitter_mean = jitter_results['jitter_mean']
    jitter_std = jitter_results['jitter_std']
    jitter_p95 = jitter_results.get('jitter_p95', np.nan)
    jitter_max = jitter_results.get('jitter_max', np.nan)
    drift_mean = jitter_results.get('drift_mean', np.nan)
    hf_jitter = jitter_results.get('hf_jitter', np.nan)
    per_frame_jitter = jitter_results.get('per_frame_jitter', np.array([]))

    if not np.isnan(jitter_mean):
        print(f"Mean Static Jitter:      {jitter_mean:.6f} m")
        print(f"Jitter Std (spatial):    {jitter_std:.6f} m")
        print(f"Jitter P95:              {jitter_p95:.6f} m")
        print(f"Jitter Max:              {jitter_max:.6f} m")
        print(f"Drift Mean:              {drift_mean:.6f} m")
        if not np.isnan(hf_jitter):
            print(f"HF Jitter (accel):       {hf_jitter:.6f} m")
        print(f"Jitter Anchors:          {jitter_results['n_anchors']}")

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

    # ── Scatter plots: per-frame metrics vs per-frame jitter ─────────────
    if len(per_frame_jitter) > 0:
        # per_frame_jitter has shape (T-1,); align with frame indices [1..T-1]
        # use the "destination" frame's accuracy/chamfer for each transition
        n_transitions = len(per_frame_jitter)
        pf_chamfer = np.array(chamfer_distances[1:n_transitions + 1])
        pf_static_acc = np.array(static_accuracies[1:n_transitions + 1])
        pf_dynamic_acc = np.array(dynamic_accuracies[1:n_transitions + 1])
        pf_motion_gap = pf_static_acc - pf_dynamic_acc

        title_tag = plot_prefix.rstrip('_') if plot_prefix else "config"

        _plot_scatter_with_regression(
            pf_chamfer, per_frame_jitter,
            xlabel="Chamfer Distance", ylabel="Jitter (m)",
            title=f"Chamfer vs Jitter — {title_tag}",
            save_path=os.path.join(out_plot_dir, f'{plot_prefix}scatter_chamfer_vs_jitter.png'),
        )
        _plot_scatter_with_regression(
            pf_static_acc, per_frame_jitter,
            xlabel="Static Accuracy", ylabel="Jitter (m)",
            title=f"Static Accuracy vs Jitter — {title_tag}",
            save_path=os.path.join(out_plot_dir, f'{plot_prefix}scatter_static_acc_vs_jitter.png'),
        )
        _plot_scatter_with_regression(
            pf_dynamic_acc, per_frame_jitter,
            xlabel="Dynamic Accuracy", ylabel="Jitter (m)",
            title=f"Dynamic Accuracy vs Jitter — {title_tag}",
            save_path=os.path.join(out_plot_dir, f'{plot_prefix}scatter_dynamic_acc_vs_jitter.png'),
        )
        _plot_scatter_with_regression(
            pf_motion_gap, per_frame_jitter,
            xlabel="Motion Gap (Static - Dynamic)", ylabel="Jitter (m)",
            title=f"Motion Gap vs Jitter — {title_tag}",
            save_path=os.path.join(out_plot_dir, f'{plot_prefix}scatter_motion_gap_vs_jitter.png'),
        )

    return {
        "mean_static_accuracy": mean_static,
        "mean_dynamic_accuracy": mean_dynamic,
        "mean_chamfer": np.mean(chamfer_distances),
        "mean_completeness": np.nanmean(completeness_scores),
        "static_accuracies": static_accuracies,
        "dynamic_accuracies": dynamic_accuracies,
        "jitter_mean": jitter_mean,
        "jitter_std": jitter_std,
        "jitter_p95": jitter_p95,
        "jitter_max": jitter_max,
        "drift_mean": drift_mean,
        "hf_jitter": hf_jitter,
        "ate": np.nanmean(ate_list) if ate_list else np.nan,
        "rpe": np.nanmean(rpe_list) if rpe_list else np.nan,
        "rot_err": np.nanmean(rot_err_list) if rot_err_list else np.nan,
        "focal_err": np.nanmean(focal_err_list) if focal_err_list else np.nan,
        "pp_err": np.nanmean(pp_err_list) if pp_err_list else np.nan,
    }


def plot_accuracy_vs_cameras(out_plot_dir, camera_counts, static_means, dynamic_means, filename):
    plt.figure()
    plt.plot(camera_counts, static_means, marker='o', color='blue', label='Static Accuracy')
    plt.plot(camera_counts, dynamic_means, marker='x', color='red', label='Dynamic Accuracy')
    plt.title('Static vs Dynamic Accuracy vs Number of Cameras')
    plt.xlabel('Number of Cameras')
    plt.ylabel('Accuracy (% < 1cm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_plot_dir, filename))
    plt.close()


def evaluate_camera_configs(base_in_dir, out_plot_dir, camera_counts, plot_prefix=""):
    results = {}
    available_counts = []
    for cam_count in camera_counts:
        in_dir = os.path.join(base_in_dir, f"{cam_count}views")
        if not os.path.isdir(in_dir):
            print(f"[INFO] Missing directory: {in_dir}, skipping.")
            continue
        metrics = evaluate_configuration(
            in_dir=in_dir,
            out_plot_dir=out_plot_dir,
            plot_prefix=f"{plot_prefix}{cam_count}views_",
        )
        if metrics is None:
            continue
        available_counts.append(cam_count)
        results[cam_count] = {
            "static": metrics["mean_static_accuracy"],
            "dynamic": metrics["mean_dynamic_accuracy"],
            "chamfer": metrics["mean_chamfer"],
            "completeness": metrics["mean_completeness"],
            "jitter_mean": metrics["jitter_mean"],
            "jitter_std": metrics["jitter_std"],
            "jitter_p95": metrics["jitter_p95"],
            "jitter_max": metrics["jitter_max"],
            "drift_mean": metrics["drift_mean"],
            "hf_jitter": metrics["hf_jitter"],
            "ate": metrics["ate"],
            "rpe": metrics["rpe"],
            "rot_err": metrics["rot_err"],
            "focal_err": metrics["focal_err"],
            "pp_err": metrics["pp_err"],
        }
    return available_counts, results


def get_selected_subject_names(args):
    if args.all:
        return SUBJECT_NAMES

    selected_codes = [
        code for code in sorted(SUBJECT_BY_CODE.keys())
        if getattr(args, f"subject_{code}")
    ]
    if not selected_codes:
        selected_codes = ["01"]  # default selection
    return [SUBJECT_BY_CODE[code] for code in selected_codes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Custom directory containing frame_*.npz files")
    parser.add_argument("--all", action="store_true", help="Evaluate all subjects (01..10).")
    for code in sorted(SUBJECT_BY_CODE.keys()):
        parser.add_argument(f"--{code}", dest=f"subject_{code}", action="store_true", help=f"Evaluate subject {code}.")
    args = parser.parse_args()

    np.random.seed(42)  # For reproducibility

    os.makedirs("plots", exist_ok=True)

    if args.input_dir:
        in_dir = args.input_dir
        out_plot_dir = in_dir.replace("aligned_outputs", "plots")
        metrics = evaluate_configuration(in_dir, out_plot_dir)
        if metrics is None:
            print(f"No aligned output files found in {in_dir}. Please run align_reconstruction_umeyama.py first.")
            return
        print(f"\nPlots saved in {out_plot_dir}/ directory.")
        return

    base_in_dir = "aligned_outputs"
    camera_counts = [2, 3, 4]
    selected_subjects = set(get_selected_subject_names(args))
    subject_dirs = [
        d for d in sorted(glob.glob(os.path.join(base_in_dir, "*")))
        if os.path.isdir(d)
    ]
    subject_dirs = [d for d in subject_dirs if os.path.basename(d) in selected_subjects]
    subject_dirs = [
        d for d in subject_dirs
        if any(os.path.isdir(os.path.join(d, f"{cam}views")) for cam in camera_counts)
    ]
    selected_codes_str = ", ".join(name.split("subject-")[1][:2] for name in sorted(selected_subjects))
    print(f"[INFO] Selected subjects: {selected_codes_str}")

    if subject_dirs:
        csv_rows = []
        all_subject_results = {}
        for subject_dir in subject_dirs:
            subject_name = os.path.basename(subject_dir)
            subject_plot_dir = os.path.join("plots", subject_name)
            print(f"\n=== Evaluating subject: {subject_name} ===")
            available_counts, results = evaluate_camera_configs(
                base_in_dir=subject_dir,
                out_plot_dir=subject_plot_dir,
                camera_counts=camera_counts,
            )
            if not available_counts:
                print(f"[WARN] No valid camera configuration outputs for {subject_name}, skipping.")
                continue

            static_means = [results[c]["static"] for c in available_counts]
            dynamic_means = [results[c]["dynamic"] for c in available_counts]
            plot_accuracy_vs_cameras(
                out_plot_dir=subject_plot_dir,
                camera_counts=available_counts,
                static_means=static_means,
                dynamic_means=dynamic_means,
                filename="accuracy_vs_cameras.png",
            )

            all_subject_results[subject_name] = results
            for cam_count in available_counts:
                res = results[cam_count]
                print(
                    f"  {cam_count} views -> "
                    f"static={res['static']:.5f}, "
                    f"dynamic={res['dynamic']:.5f}, "
                    f"jitter={res['jitter_mean']:.6f}, "
                    f"ate={res['ate']:.4f}, "
                    f"rpe={res['rpe']:.4f}, "
                    f"rot_err={res['rot_err']:.4f}, "
                    f"focal_err={res['focal_err']:.5f}, "
                    f"pp_err={res['pp_err']:.3f}"
                )
                csv_rows.append({
                    "subject": subject_name,
                    "views": cam_count,
                    "chamfer": res["chamfer"],
                    "completeness": res["completeness"],
                    "static_acc": res["static"],
                    "dynamic_acc": res["dynamic"],
                    "jitter_mean": res["jitter_mean"],
                    "jitter_std": res["jitter_std"],
                    "jitter_p95": res["jitter_p95"],
                    "jitter_max": res["jitter_max"],
                    "drift_mean": res["drift_mean"],
                    "hf_jitter": res["hf_jitter"],
                    "ate": res["ate"],
                    "rpe": res["rpe"],
                    "rot_err": res["rot_err"],
                    "focal_err": res["focal_err"],
                    "pp_err": res["pp_err"],
                })
            print(f"Plots saved in {subject_plot_dir}/ directory.")

        if not all_subject_results:
            print("No valid aligned outputs found. Please run align_reconstruction_umeyama.py first.")
            return

        print("\n=== Cross-subject average metrics per view count ===")
        for cam_count in camera_counts:
            static_vals = []
            dynamic_vals = []
            ate_vals = []
            rot_vals = []
            for subject_name, subject_results in all_subject_results.items():
                if cam_count not in subject_results:
                    continue
                res = subject_results[cam_count]
                if not np.isnan(res["static"]): static_vals.append(res["static"])
                if not np.isnan(res["dynamic"]): dynamic_vals.append(res["dynamic"])
                if not np.isnan(res["ate"]): ate_vals.append(res["ate"])
                if not np.isnan(res["rot_err"]): rot_vals.append(res["rot_err"])

            mean_static = float(np.mean(static_vals)) if static_vals else np.nan
            mean_dynamic = float(np.mean(dynamic_vals)) if dynamic_vals else np.nan
            mean_ate = float(np.mean(ate_vals)) if ate_vals else np.nan
            mean_rot = float(np.mean(rot_vals)) if rot_vals else np.nan

            print(
                f"{cam_count} views -> "
                f"mean static={mean_static:.5f}, "
                f"mean dynamic={mean_dynamic:.5f}, "
                f"mean ate={mean_ate:.4f}, "
                f"mean rot={mean_rot:.4f}, "
                f"subjects={len(static_vals)}/{len(subject_dirs)}"
            )

        # Write to CSV
        csv_file = "evaluation_metrics.csv"
        csv_fieldnames = [
            "subject", "views", "chamfer", "completeness", "static_acc", "dynamic_acc",
            "jitter_mean", "jitter_std", "jitter_p95", "jitter_max", "drift_mean", "hf_jitter",
            "ate", "rpe", "rot_err", "focal_err", "pp_err"
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[INFO] All metrics saved to {csv_file}")
        return

    out_plot_dir = "plots"
    available_counts, results = evaluate_camera_configs(
        base_in_dir=base_in_dir,
        out_plot_dir=out_plot_dir,
        camera_counts=camera_counts,
    )
    if not available_counts:
        fallback_metrics = evaluate_configuration(base_in_dir, out_plot_dir)
        if fallback_metrics is None:
            print("No valid aligned outputs found. Please run align_reconstruction_umeyama.py first.")
            return
        print(f"\nPlots saved in {out_plot_dir}/ directory.")
        return

    static_means = [results[c]["static"] for c in available_counts]
    dynamic_means = [results[c]["dynamic"] for c in available_counts]
    plot_accuracy_vs_cameras(
        out_plot_dir=out_plot_dir,
        camera_counts=available_counts,
        static_means=static_means,
        dynamic_means=dynamic_means,
        filename="accuracy_vs_cameras.png",
    )

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
