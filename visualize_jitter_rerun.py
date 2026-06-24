#!/usr/bin/env python3
"""
Visualize the exact per-view persistent anchors used by compute_static_jitter().

This script is intentionally read-only with respect to reconstruction outputs:
it loads existing frame_*.npz files, reproduces the jitter fusion/sampling logic,
and logs the result to Rerun.
"""

import argparse
import glob
import os

import cv2
import numpy as np

from vggt.utils.umeyama_alignment import apply_similarity_transform
from vggt.utils.gt import build_gt_validity_masks
from vggt.utils.camera_utils import discover_view_name
from eval_config import DATASETS, CONF_PERCENTILE


def _load_frames(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "frame_*.npz")))
    if not files:
        raise FileNotFoundError(f"No frame_*.npz files found in {input_dir}")
    return files


def _infer_dataset_root(path, dataset_type):
    for subject_name in DATASETS[dataset_type].get("subject_names", []):
        if subject_name in path:
            return os.path.join(DATASETS[dataset_type]["root"], subject_name)
    return None


def _spatial_pointmaps(pmaps, masks):
    if pmaps.ndim == 4:
        return pmaps
    if pmaps.ndim != 3:
        raise ValueError(f"Expected pointmaps with 3 or 4 dims, got {pmaps.shape}")

    h_mask, w_mask = masks.shape[1], masks.shape[2]
    n = pmaps.shape[1]
    h_pm = int(np.round(np.sqrt(n * h_mask / float(w_mask))))
    w_pm = n // h_pm
    if h_pm * w_pm != n:
        raise ValueError(f"Cannot infer spatial pointmap shape from {pmaps.shape}")
    return pmaps.reshape(pmaps.shape[0], h_pm, w_pm, 3)


def _align_pointmaps(data):
    pmaps = _spatial_pointmaps(data["pointmaps"], data["masks_2d"]).astype(np.float32)
    if all(k in data for k in ("scale", "R", "tr")):
        s_val, r_val, tr_val = data["scale"], data["R"], data["tr"]
        aligned = np.empty_like(pmaps)
        for vi in range(pmaps.shape[0]):
            aligned[vi] = apply_similarity_transform(
                pmaps[vi].reshape(-1, 3), s_val, r_val, tr_val
            ).reshape(pmaps[vi].shape)
        return aligned
    return pmaps


def _resize_mask_stack(mask_stack, v_count, height, width, fill=True):
    if mask_stack is None:
        return np.full((v_count, height, width), fill, dtype=bool)
    resized = np.empty((v_count, height, width), dtype=bool)
    for vi in range(v_count):
        mask = mask_stack[vi]
        if mask is None:
            resized[vi] = fill
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        resized[vi] = mask
    return resized


def _build_validity_masks(data, dataset_root, dataset_type, height, width):
    if dataset_root is None:
        return None
    v_count = data["pointmaps"].shape[0]
    frame_id = int(data["frame_idx"]) if "frame_idx" in data else 0
    if "view_names" in data:
        view_names = (data["view_names"].tolist()
                      if hasattr(data["view_names"], "tolist")
                      else list(data["view_names"]))
    elif "Ks" in data:
        view_names = [
            discover_view_name(dataset_root, k, dataset_type=dataset_type)
            for k in data["Ks"]
        ]
    else:
        return None

    if any(v is None for v in view_names):
        return None

    vmasks = build_gt_validity_masks(
        frame_id, view_names, dataset_root,
        target_hw=(height, width), dataset_type=dataset_type,
    )
    return np.array([
        vmask if vmask is not None else np.ones((height, width), dtype=bool)
        for vmask in vmasks
    ], dtype=bool)


def _valid_masks_like_compute_static_jitter(
        pointmaps_per_frame,
        masks_per_frame,
        validity_masks_per_frame=None,
        confidences_per_frame=None,
        conf_percentile=0.5,
):
    first_pm = pointmaps_per_frame[0]
    v_count, height, width, _ = first_pm.shape

    valid_masks = []
    for t, (pm_t, mk_t) in enumerate(zip(pointmaps_per_frame, masks_per_frame)):
        static_mask = _resize_mask_stack(mk_t, v_count, height, width, fill=False)
        gt_valid = (
            _resize_mask_stack(validity_masks_per_frame[t], v_count, height, width, fill=True)
            if validity_masks_per_frame is not None
            else np.ones((v_count, height, width), dtype=bool)
        )
        if confidences_per_frame is not None:
            conf_t = confidences_per_frame[t]
            conf_mask = np.zeros((v_count, height, width), dtype=bool)
            for vi in range(v_count):
                conf = conf_t[vi]
                if conf.shape != (height, width):
                    conf = cv2.resize(
                        conf.astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                finite_conf = np.isfinite(conf)
                if np.any(finite_conf):
                    thr = np.percentile(conf[finite_conf], 100 * (1 - conf_percentile))
                    conf_mask[vi] = finite_conf & (conf > thr)
        else:
            conf_mask = np.ones((v_count, height, width), dtype=bool)
        finite = np.isfinite(pm_t).all(axis=-1)
        nonzero = np.linalg.norm(pm_t, axis=-1) > 1e-8
        valid_masks.append(static_mask & gt_valid & conf_mask & finite & nonzero)

    return valid_masks


def _sample_anchor_trajectories(pointmaps_per_frame, valid_masks, n_anchors, seed):
    v_count, height, width = valid_masks[0].shape
    anchor_mask = np.ones((v_count, height, width), dtype=bool)
    for mask_t in valid_masks:
        anchor_mask &= mask_t

    flat_indices = np.where(anchor_mask.ravel())[0]
    if len(flat_indices) < 2:
        raise RuntimeError("Fewer than 2 persistent jitter anchors were found.")

    rng = np.random.default_rng(seed)
    n_sample = min(n_anchors, len(flat_indices))
    sampled_indices = rng.choice(flat_indices, size=n_sample, replace=False)
    views, ys, xs = np.unravel_index(sampled_indices, (v_count, height, width))

    stacked = np.stack(pointmaps_per_frame)
    trajectories = stacked[:, views, ys, xs]
    return anchor_mask, sampled_indices, views, ys, xs, trajectories


def _compute_jitter_stats(trajectories):
    displacements = np.linalg.norm(trajectories[1:] - trajectories[:-1], axis=-1)
    per_anchor_mean = np.mean(displacements, axis=0)
    drift = np.linalg.norm(trajectories[-1] - trajectories[0], axis=-1)
    stats = {
        "jitter_mean": float(np.mean(displacements)),
        "jitter_std": float(np.std(per_anchor_mean)),
        "jitter_p95": float(np.percentile(per_anchor_mean, 95)),
        "jitter_max": float(np.max(per_anchor_mean)),
        "drift_mean": float(np.mean(drift)),
    }
    if trajectories.shape[0] >= 3:
        accel = np.linalg.norm(
            trajectories[2:] - 2 * trajectories[1:-1] + trajectories[:-2],
            axis=-1,
        )
        stats["hf_jitter"] = float(np.mean(accel))
    else:
        stats["hf_jitter"] = np.nan
    return stats


def _sample_for_logging(points, colors=None, max_points=100000, seed=0):
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    if colors is None:
        return points[idx], None
    return points[idx], colors[idx]


def _finite_points(points):
    return points[np.isfinite(points).all(axis=1)]


def _log_line_strips(rr, entity, trajectories, color, radius):
    try:
        rr.log(entity, rr.LineStrips3D(strips=trajectories, colors=color, radii=radius))
        return
    except Exception:
        pass
    try:
        rr.log(entity, rr.LineStrips3D(trajectories, colors=color, radii=radius))
    except Exception as exc:
        print(f"[WARN] Could not log LineStrips3D with this Rerun version: {exc}")


def _set_frame_time(rr, frame_id):
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_id)
    else:
        rr.set_time("frame", sequence=frame_id)


def _confidence_filtered_points(data, aligned_pmaps, conf_percentile):
    if "pointmaps_confs" not in data:
        return None
    confs = data["pointmaps_confs"]
    if confs.ndim == 2:
        confs = confs.reshape(aligned_pmaps.shape[:3])
    points = []
    for vi in range(aligned_pmaps.shape[0]):
        conf_i = confs[vi].reshape(-1)
        pts_i = aligned_pmaps[vi].reshape(-1, 3)
        thr_i = np.percentile(conf_i, 100 * (1 - conf_percentile))
        valid = (conf_i > thr_i) & np.isfinite(pts_i).all(axis=1)
        points.append(pts_i[valid])
    if not points:
        return None
    return np.concatenate(points, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing frame_*.npz files.")
    parser.add_argument("--n_anchors", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cloud_points", type=int, default=100000)
    parser.add_argument("--max_track_lines", type=int, default=1000)
    parser.add_argument("--conf_percentile", type=float, default=CONF_PERCENTILE)
    parser.add_argument("--log_conf_filtered", action="store_true",
                        help="Also log the confidence-filtered aligned cloud for comparison.")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional .rrd path. If omitted, stream/spawn Rerun.")
    parser.add_argument("--connect", type=str, default=None,
                        help="Optional Rerun address, e.g. 127.0.0.1:9876.")
    parser.add_argument("--spawn", action="store_true",
                        help="Spawn a local Rerun viewer if supported.")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Optional dataset subject root for GT validity masks.")
    parser.add_argument("--data", type=str, choices=["dex-ycb", "hi4d"], default="dex-ycb")
    args = parser.parse_args()

    try:
        import rerun as rr
    except ImportError as exc:
        raise SystemExit(
            "The rerun Python package is not available in this environment. "
            "Run this script from the same environment you use for the existing "
            "Rerun reconstruction scripts."
        ) from exc

    files = _load_frames(args.input_dir)
    dataset_root = args.dataset_root or _infer_dataset_root(args.input_dir, args.data)
    if dataset_root is None:
        print("[jitter-viz] No dataset root found; GT validity masks will not be used.")
    else:
        print(f"[jitter-viz] Using GT validity masks from {dataset_root}")
    pointmaps_per_frame = []
    masks_per_frame = []
    validity_masks_per_frame = []
    confidences_per_frame = []
    frame_ids = []
    loaded_npz = []

    for path in files:
        data = np.load(path)
        if "pointmaps" not in data or "masks_2d" not in data:
            print(f"[WARN] Skipping {path}: missing pointmaps or masks_2d")
            continue
        aligned_pmaps = _align_pointmaps(data)
        pointmaps_per_frame.append(aligned_pmaps)
        masks_per_frame.append(data["masks_2d"].astype(bool))
        if "pointmaps_confs" in data:
            confs = data["pointmaps_confs"]
            if confs.ndim == 2:
                confs = confs.reshape(aligned_pmaps.shape[:3])
            confidences_per_frame.append(confs)
        validity = _build_validity_masks(
            data, dataset_root, args.data,
            aligned_pmaps.shape[1], aligned_pmaps.shape[2],
        )
        if validity is not None:
            validity_masks_per_frame.append(validity)
        frame_ids.append(int(data["frame_idx"]) if "frame_idx" in data else len(frame_ids))
        loaded_npz.append(data)

    if len(pointmaps_per_frame) < 2:
        raise RuntimeError("Need at least two valid frames to visualize jitter.")

    validity_for_jitter = (
        validity_masks_per_frame
        if len(validity_masks_per_frame) == len(pointmaps_per_frame)
        else None
    )
    confidences_for_jitter = (
        confidences_per_frame
        if len(confidences_per_frame) == len(pointmaps_per_frame)
        else None
    )
    valid_masks = _valid_masks_like_compute_static_jitter(
        pointmaps_per_frame,
        masks_per_frame,
        validity_for_jitter,
        confidences_for_jitter,
        args.conf_percentile,
    )
    anchor_mask, sampled_indices, views, ys, xs, trajectories = _sample_anchor_trajectories(
        pointmaps_per_frame, valid_masks, args.n_anchors, args.seed
    )
    stats = _compute_jitter_stats(trajectories)

    print(f"[jitter-viz] Loaded {len(pointmaps_per_frame)} frames from {args.input_dir}")
    print(f"[jitter-viz] Per-view resolution: {valid_masks[0].shape[2]}x{valid_masks[0].shape[1]}")
    if confidences_for_jitter is not None:
        print(f"[jitter-viz] Confidence filtering: retaining top {100 * args.conf_percentile:.1f}% per frame/view")
    else:
        print("[jitter-viz] Confidence filtering: disabled (no complete pointmaps_confs found)")
    print(f"[jitter-viz] Per-view persistent anchors: "
          f"{', '.join(str(int(c)) for c in anchor_mask.reshape(anchor_mask.shape[0], -1).sum(axis=1))}")
    print(f"[jitter-viz] Persistent potential anchors: {int(anchor_mask.sum()):,}")
    print(f"[jitter-viz] Sampled anchors: {len(sampled_indices):,}")
    for key, val in stats.items():
        print(f"[jitter-viz] {key}: {val:.6f}")

    rr.init("vggt4d_jitter_debug", spawn=args.spawn)
    if args.save:
        rr.save(args.save)
        print(f"[rerun] saving recording to {args.save}")
    elif args.connect:
        try:
            rr.connect_grpc(args.connect)
        except Exception:
            rr.connect(args.connect)
        print(f"[rerun] streaming to {args.connect}")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    track_count = min(args.max_track_lines, trajectories.shape[1])
    _log_line_strips(
        rr,
        "world/jitter/anchor_tracks",
        trajectories[:, :track_count].transpose(1, 0, 2),
        color=[255, 220, 0],
        radius=0.0015,
    )

    for i, (frame_id, valid_mask, data, aligned_pmaps) in enumerate(
        zip(frame_ids, valid_masks, loaded_npz, pointmaps_per_frame)
    ):
        _set_frame_time(rr, frame_id)

        valid_points = _finite_points(aligned_pmaps[valid_mask].reshape(-1, 3))
        valid_points_log, _ = _sample_for_logging(
            valid_points, max_points=args.max_cloud_points, seed=args.seed + i
        )
        rr.log(
            "world/jitter/exact_static_valid_pointcloud",
            rr.Points3D(valid_points_log, colors=[70, 170, 255], radii=0.002),
        )

        anchor_points = trajectories[i]
        rr.log(
            "world/jitter/sampled_anchors",
            rr.Points3D(anchor_points, colors=[255, 40, 40], radii=0.006),
        )

        raw_points = _finite_points(aligned_pmaps.reshape(-1, 3))
        raw_points_log, _ = _sample_for_logging(
            raw_points, max_points=args.max_cloud_points, seed=args.seed + 1000 + i
        )
        rr.log(
            "world/jitter/aligned_raw_pointmaps_all_views",
            rr.Points3D(raw_points_log, colors=[120, 120, 120], radii=0.001),
        )

        if args.log_conf_filtered:
            conf_points = _confidence_filtered_points(data, aligned_pmaps, args.conf_percentile)
            if conf_points is not None:
                conf_points_log, _ = _sample_for_logging(
                    conf_points, max_points=args.max_cloud_points, seed=args.seed + 2000 + i
                )
                rr.log(
                    "world/jitter/confidence_filtered_comparison",
                    rr.Points3D(conf_points_log, colors=[0, 220, 120], radii=0.002),
                )

    print("[jitter-viz] Done. In Rerun, inspect:")
    print("  world/jitter/exact_static_valid_pointcloud      blue: exact candidate cloud used for jitter")
    print("  world/jitter/sampled_anchors                    red: anchors at current frame")
    print("  world/jitter/anchor_tracks                      yellow: sampled anchor trajectories")
    print("  world/jitter/aligned_raw_pointmaps_all_views    gray: raw aligned pointmap inputs")
    if args.log_conf_filtered:
        print("  world/jitter/confidence_filtered_comparison     green: confidence-filtered comparison")


if __name__ == "__main__":
    main()
