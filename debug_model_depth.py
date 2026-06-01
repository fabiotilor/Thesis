#!/usr/bin/env python3
"""
Analyze raw model output per-frame to check depth consistency.
Loads saved NPZ files and reports pointmap statistics per frame.

Usage:
    python debug_model_depth.py --npz-dir aligned_outputs/baseline/hi4d/pair00/dance00/2views
"""
import os
import glob
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir", type=str, required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.npz_dir, "frame_*.npz")))
    if not paths:
        print(f"No NPZ files found in {args.npz_dir}")
        return

    print(f"{'='*90}")
    print(f"Model Depth Consistency Analysis — {len(paths)} frames")
    print(f"Dir: {args.npz_dir}")
    print(f"{'='*90}\n")

    # Collect per-frame stats
    all_scales = []
    all_centroids = []
    all_gt_centroids = []

    print(f"{'Frame':>6} {'Scale':>8} {'#Pts':>8} "
          f"{'Raw Centroid (x,y,z)':>35} {'Raw StdDev':>12} "
          f"{'GT Centroid (x,y,z)':>35} {'GT StdDev':>12}")
    print("-" * 130)

    for p in paths:
        data = np.load(p, allow_pickle=True)
        frame_idx = int(data["frame_idx"])
        scale = float(data["scale"])
        all_scales.append(scale)

        # Raw pointmaps (before alignment)
        pm = data.get("pointmaps")  # (V, H, W, 3)
        confs = data.get("pointmaps_confs")  # (V, H, W)

        if pm is not None:
            # Flatten and filter zero points (masked out)
            pm_flat = pm.reshape(-1, 3)
            if confs is not None:
                conf_flat = confs.ravel()
                valid = (conf_flat > 0.01) & (np.linalg.norm(pm_flat, axis=1) > 0.01)
            else:
                valid = np.linalg.norm(pm_flat, axis=1) > 0.01

            pts_valid = pm_flat[valid]
            n_pts = len(pts_valid)

            if n_pts > 0:
                centroid = pts_valid.mean(axis=0)
                std = pts_valid.std()
                all_centroids.append(centroid)
            else:
                centroid = np.zeros(3)
                std = 0.0
        else:
            n_pts = 0
            centroid = np.zeros(3)
            std = 0.0

        # GT stats
        gt_pts = data.get("gt_pts")
        if gt_pts is not None and len(gt_pts) > 0:
            gt_centroid = gt_pts.mean(axis=0)
            gt_std = gt_pts.std()
            all_gt_centroids.append(gt_centroid)
        else:
            gt_centroid = np.zeros(3)
            gt_std = 0.0

        print(f"{frame_idx:6d} {scale:8.4f} {n_pts:8d} "
              f"[{centroid[0]:7.3f}, {centroid[1]:7.3f}, {centroid[2]:7.3f}] {std:12.4f} "
              f"[{gt_centroid[0]:7.3f}, {gt_centroid[1]:7.3f}, {gt_centroid[2]:7.3f}] {gt_std:12.4f}")

    # Summary
    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")

    scales = np.array(all_scales)
    print(f"  Scale: mean={scales.mean():.4f}  std={scales.std():.4f}  "
          f"min={scales.min():.4f}  max={scales.max():.4f}  "
          f"ratio(max/min)={scales.max()/scales.min():.2f}x")

    if all_centroids:
        centroids = np.array(all_centroids)
        centroid_drifts = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        print(f"\n  Raw centroid drift (frame-to-frame):")
        print(f"    mean={centroid_drifts.mean():.4f}  max={centroid_drifts.max():.4f}")
        print(f"    centroid X range: [{centroids[:,0].min():.3f}, {centroids[:,0].max():.3f}]")
        print(f"    centroid Y range: [{centroids[:,1].min():.3f}, {centroids[:,1].max():.3f}]")
        print(f"    centroid Z range: [{centroids[:,2].min():.3f}, {centroids[:,2].max():.3f}]")

    if all_gt_centroids:
        gt_centroids = np.array(all_gt_centroids)
        gt_drifts = np.linalg.norm(np.diff(gt_centroids, axis=0), axis=1)
        print(f"\n  GT centroid drift (frame-to-frame):")
        print(f"    mean={gt_drifts.mean():.4f}  max={gt_drifts.max():.4f}")

    # Per-view analysis
    print(f"\n  Per-view depth analysis (first frame):")
    data0 = np.load(paths[0], allow_pickle=True)
    pm0 = data0.get("pointmaps")
    confs0 = data0.get("pointmaps_confs")
    if pm0 is not None:
        for v in range(pm0.shape[0]):
            pts_v = pm0[v].reshape(-1, 3)
            conf_v = confs0[v].ravel() if confs0 is not None else np.ones(len(pts_v))
            valid_v = (conf_v > 0.01) & (np.linalg.norm(pts_v, axis=1) > 0.01)
            pts_vv = pts_v[valid_v]
            if len(pts_vv) > 0:
                depths = np.linalg.norm(pts_vv, axis=1)
                print(f"    View {v}: {len(pts_vv)} pts, "
                      f"depth range=[{depths.min():.3f}, {depths.max():.3f}], "
                      f"centroid=[{pts_vv.mean(0)[0]:.3f}, {pts_vv.mean(0)[1]:.3f}, {pts_vv.mean(0)[2]:.3f}]")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
