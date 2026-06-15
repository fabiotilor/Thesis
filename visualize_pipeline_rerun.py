#!/usr/bin/env python3
"""
Visualise the 3D / 4D reconstruction-evaluation pipeline in Rerun.

This script walks through each stage of the evaluation pipeline — from raw
model reconstruction through confidence filtering, correspondence
construction, and Umeyama alignment — and logs every intermediate result to
Rerun so that a user can scrub through the timeline and take publication-
quality screenshots.

Expected input layout
---------------------
  --data_dir      Root of the dataset for one subject / sequence.
                  • DEX-YCB:  …/<subject_name>/        (contains view_XX/)
                  • Hi4D:     …/<pairXX>/<actionXX>/   (contains images/, cameras/, …)

  --model_output  Directory produced by `align_reconstruction_umeyama.py`
                  containing per-frame `frame_XXXXXX.npz` files.  Each file
                  stores: pointmaps, pointmaps_confs, gt_pts, aligned_pts,
                  Ks, R_ts, masks_2d, est_poses, est_intrinsics,
                  scale, R, tr, min_conf_thr, conf_percentile, frame_idx.

Usage examples
--------------
# Resolve paths automatically from the standard folder layout:
#   aligned_outputs/pi3/<dataset>/<strategy>/<sequence>/<nviews>views/
python visualize_pipeline_rerun.py \
    --data_dir /home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754 \
    --output_root aligned_outputs/pi3 \
    --dataset dexycb \
    --strategy baseline \
    --nviews 4 \
    --frame 14

# Visualise strategy-2 output (4D pipeline uses strategy2 folder automatically):
python visualize_pipeline_rerun.py \
    --data_dir /home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754 \
    --output_root aligned_outputs/pi3 \
    --dataset dexycb \
    --strategy strategy2 \
    --nviews 4

# Or supply a fully-explicit path with --model_output (still supported):
python visualize_pipeline_rerun.py \
    --data_dir /home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754 \
    --model_output aligned_outputs/pi3/dex-ycb/baseline/20200709-subject-01__20200709_141754/4views \
    --dataset dexycb
"""

import os
import sys
import glob
import argparse
from typing import Optional

import numpy as np
import cv2
import rerun as rr
import matplotlib
import matplotlib.cm as cm

# ── Project imports ───────────────────────────────────────────────────────────
from pi3.utils.umeyama_alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
)
from pi3.utils.gt import (
    load_gt_params,
    build_gt_pointcloud,
    build_gt_validity_masks,
    get_static_correspondences,
    DEPTH_MAX_M,
    DEPTH_SCALE,
)
from pi3.utils.camera_utils import discover_view_name, get_rgb_path
from pi3.utils.alignment_4d import (
    normalize_spatial_dims,
    normalize_array,
    strategy2_hierarchical,
    solve_final_gt_registration,
)
from pi3.utils.optical_flow import compute_static_mask

from eval_config import (
    CONF_PERCENTILE,
    DATASETS,
    RERUN_EYE_UP,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _viridis_colors(values: np.ndarray) -> np.ndarray:
    """Map scalar *values* in [0, 1] to RGBA uint8 via the viridis colourmap."""
    cmap = matplotlib.colormaps["viridis"]
    rgba = cmap(values)  # (N, 4) float in [0, 1]
    return (rgba * 255).astype(np.uint8)


def _parse_steps(raw: str) -> set[int]:
    """Parse --steps value into a set of step integers.

    Accepted forms:
      '3d'        → {0, 1, 2, 3, 4, 5, 6}
      '4d'        → {7, 8, 9, 10}
      'all'       → {0, …, 10}
      '0,1,2'     → {0, 1, 2}
      '6,9,10'    → {6, 9, 10}
    """
    raw = raw.strip().lower()
    if raw == "all":
        return set(range(11))
    if raw == "3d":
        return set(range(7))
    if raw == "4d":
        return {7, 8, 9, 10}
    try:
        return {int(s.strip()) for s in raw.split(",")}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --steps value '{raw}'. "
            "Use 'all', '3d', '4d', or a comma-separated list of step numbers 0-10."
        )


def _load_frame_npz(path: str) -> dict:
    """Load an .npz frame file and return its contents as a plain dict."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _recover_view_names(
        data: dict, dataset_root: str, dataset_type: str
) -> list[str]:
    """Recover per-view folder names from intrinsic matrices stored in the NPZ."""
    if "view_names" in data:
        return list(data["view_names"])
    ks = data["Ks"]
    return [
        discover_view_name(dataset_root, k, dataset_type=dataset_type)
        for k in ks
    ]


def _collect_frame_paths(model_output: str) -> list[str]:
    """Return sorted list of frame_*.npz paths in *model_output*."""
    return sorted(
        glob.glob(os.path.join(model_output, "frame_*.npz")),
        key=lambda x: int(
            os.path.basename(x).replace("frame_", "").replace(".npz", "")
        ),
    )


def _extract_rgb_for_views(
        data: dict,
        view_names: list[str],
        frame_idx: int,
        dataset_root: str,
        dataset_type: str,
) -> list[Optional[np.ndarray]]:
    """Load RGB images (as uint8 HWC arrays) for each view at *frame_idx*."""
    images: list[Optional[np.ndarray]] = []
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        rgb_path = get_rgb_path(view_dir, frame_idx, dataset_type=dataset_type)
        if rgb_path and os.path.exists(rgb_path):
            bgr = cv2.imread(rgb_path)
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None)
        else:
            images.append(None)
    return images


# ═══════════════════════════════════════════════════════════════════════════════
#  3D Pipeline  (Steps 0 – 6)
# ═══════════════════════════════════════════════════════════════════════════════

def log_raw_pointcloud(
        data: dict,
        view_names: list[str],
        frame_idx: int,
        dataset_root: str,
        dataset_type: str,
) -> None:
    """Step 0 — Log the raw model point cloud with original RGB colours."""
    rr.set_time("pipeline_step", sequence=0)

    V, H, W = normalize_spatial_dims(data)
    pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
    pts_all = pm.reshape(-1, 3)

    # Attempt to build per-point RGB from the input images.
    rgbs = _extract_rgb_for_views(data, view_names, frame_idx, dataset_root, dataset_type)
    color_parts: list[np.ndarray] = []
    for v in range(V):
        if rgbs[v] is not None:
            img = cv2.resize(rgbs[v], (W, H), interpolation=cv2.INTER_LINEAR)
            color_parts.append(img.reshape(-1, 3))
        else:
            color_parts.append(np.full((H * W, 3), 255, dtype=np.uint8))

    colors = np.concatenate(color_parts, axis=0)

    # Log raw point cloud coloured by RGB (or white if no images available)
    rr.log(
        "pipeline_3d/raw_pointcloud",
        rr.Points3D(positions=pts_all, colors=colors, radii=0.002),
    )
    print("  [✓] Step 0 — raw_reconstruction")


def log_gt_validity_mask(
        data: dict,
        view_names: list[str],
        frame_idx: int,
        dataset_root: str,
        dataset_type: str,
) -> list[Optional[np.ndarray]]:
    """Step 1 — Log GT depth image and binary validity mask.

    Returns the list of validity masks (one per view) for downstream use.
    """
    rr.set_time("pipeline_step", sequence=1)

    depth_max = DATASETS.get(dataset_type, {}).get("depth_max_m", DEPTH_MAX_M) or DEPTH_MAX_M
    V, H, W = normalize_spatial_dims(data)
    validity_masks = build_gt_validity_masks(
        frame_idx, view_names, dataset_root,
        depth_max_m=depth_max,
        target_hw=(H, W),
        dataset_type=dataset_type,
    )

    # For visualisation, stack the first view's depth + mask side by side.
    for v_idx, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, vname)

        if dataset_type == "dexycb" or dataset_type == "dex-ycb":
            depth_path = os.path.join(view_dir, "depth", f"{frame_idx:05d}.png")
            if os.path.exists(depth_path):
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                depth_m = depth_raw * DEPTH_SCALE
                # Normalise to [0, 1] for visualisation
                d_vis = depth_m.copy()
                d_vis[d_vis <= 0] = np.nan
                d_min = np.nanmin(d_vis) if np.any(~np.isnan(d_vis)) else 0.0
                d_max = np.nanmax(d_vis) if np.any(~np.isnan(d_vis)) else 1.0
                d_norm = np.nan_to_num((d_vis - d_min) / max(d_max - d_min, 1e-6), nan=0.0)
                depth_color = (matplotlib.colormaps["turbo"](d_norm)[:, :, :3] * 255).astype(np.uint8)

                # Log the colourised GT depth map
                rr.log(
                    "pipeline_3d/gt_depth_image",
                    rr.Image(depth_color),
                )

        # Log the binary validity mask (white = valid, black = invalid)
        mask = validity_masks[v_idx]
        if mask is not None:
            mask_vis = (mask.astype(np.uint8) * 255)
            if mask_vis.shape != (H, W):
                mask_vis = cv2.resize(mask_vis, (W, H), interpolation=cv2.INTER_NEAREST)
            rr.log(
                "pipeline_3d/gt_validity_mask",
                rr.Image(mask_vis),
            )
        # Only log the first view for the side-by-side figure panel.
        break

    print("  [✓] Step 1 — gt_validity_mask")
    return validity_masks


def log_masked_pointcloud(
        data: dict,
        validity_masks: list[Optional[np.ndarray]],
        view_names: list[str],
        frame_idx: int,
        dataset_root: str,
        dataset_type: str,
        do_log: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 2 — Log the point cloud after applying the GT validity mask.

    Returns (masked_pts, masked_colors, masked_confs) for subsequent steps.
    """
    if do_log:
        rr.set_time("pipeline_step", sequence=2)

    V, H, W = normalize_spatial_dims(data)
    pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)

    rgbs = _extract_rgb_for_views(data, view_names, frame_idx, dataset_root, dataset_type)
    conf = (
        normalize_array(data["pointmaps_confs"], V, H, W)
        if "pointmaps_confs" in data
        else None
    )

    pts_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []

    for v in range(V):
        mask = validity_masks[v] if validity_masks[v] is not None else np.ones((H, W), dtype=bool)
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        p = pm[v][mask]
        pts_parts.append(p)

        if rgbs[v] is not None:
            img = cv2.resize(rgbs[v], (W, H), interpolation=cv2.INTER_LINEAR)
            col_parts.append(img[mask])
        else:
            col_parts.append(np.full((len(p), 3), 255, dtype=np.uint8))

        if conf is not None:
            conf_parts.append(conf[v][mask])

    masked_pts = np.concatenate(pts_parts, axis=0)
    masked_colors = np.concatenate(col_parts, axis=0)
    masked_confs = np.concatenate(conf_parts, axis=0) if conf_parts else np.ones(len(masked_pts))

    if do_log:
        # Log masked point cloud with original RGB colouring
        rr.log(
            "pipeline_3d/masked_pointcloud",
            rr.Points3D(positions=masked_pts, colors=masked_colors, radii=0.002),
        )
        print("  [✓] Step 2 — masked_pointcloud")
    return masked_pts, masked_colors, masked_confs


def log_confidence_coloring(
        masked_pts: np.ndarray,
        masked_confs: np.ndarray,
) -> None:
    """Step 3 — Log the masked point cloud coloured by confidence (viridis)."""
    rr.set_time("pipeline_step", sequence=3)

    c_min, c_max = masked_confs.min(), masked_confs.max()
    c_norm = (masked_confs - c_min) / max(c_max - c_min, 1e-8)
    colors = _viridis_colors(c_norm)

    # Log confidence-coloured point cloud (viridis: dark purple → yellow)
    rr.log(
        "pipeline_3d/confidence_colored",
        rr.Points3D(positions=masked_pts, colors=colors, radii=0.002),
    )
    print("  [✓] Step 3 — confidence_coloring")


def log_confidence_filtered(
        masked_pts: np.ndarray,
        masked_colors: np.ndarray,
        masked_confs: np.ndarray,
) -> np.ndarray:
    """Step 4 — Log only top-50 % confidence points, coloured by RGB.

    Returns the filtered point cloud for subsequent steps.
    """
    rr.set_time("pipeline_step", sequence=4)

    thr = np.quantile(masked_confs, 1.0 - CONF_PERCENTILE)
    keep = masked_confs > thr
    filtered_pts = masked_pts[keep]
    filtered_cols = masked_colors[keep]

    # Log top-50 % confidence points to show spatial effect of filtering
    rr.log(
        "pipeline_3d/confidence_filtered",
        rr.Points3D(positions=filtered_pts, colors=filtered_cols, radii=0.002),
    )
    print(f"  [✓] Step 4 — confidence_filtered  ({len(filtered_pts):,} / {len(masked_pts):,} kept)")
    return filtered_pts


def log_correspondences(
        data: dict,
        view_names: list[str],
        frame_idx: int,
        dataset_root: str,
        dataset_type: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Step 5 — Visualise 3D-3D correspondences between estimated and GT.

    Returns (src_corr, dst_corr) for use in the Umeyama step.
    """
    rr.set_time("pipeline_step", sequence=5)

    V, H, W = normalize_spatial_dims(data)
    pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
    conf = (
        normalize_array(data["pointmaps_confs"], V, H, W)
        if "pointmaps_confs" in data
        else None
    )

    pts3d_list = [pm[v] for v in range(V)]
    confs_list = [conf[v] for v in range(V)] if conf is not None else [np.ones((H, W)) for _ in range(V)]

    src_corr, dst_corr = get_static_correspondences(
        frame_idx, view_names, pts3d_list, confs_list, dataset_root,
        conf_percentile=CONF_PERCENTILE,
        use_static_mask=False,
        dataset_type=dataset_type,
    )

    if src_corr is None or dst_corr is None or len(src_corr) < 3:
        print("  [WARN] Step 5 — too few correspondences, skipping visualisation")
        return None, None

    # Log estimated points in blue
    rr.log(
        "pipeline_3d/correspondences/estimated",
        rr.Points3D(positions=src_corr, colors=[0, 100, 255], radii=0.0015),
    )

    # Log ground-truth points in green
    rr.log(
        "pipeline_3d/correspondences/gt",
        rr.Points3D(positions=dst_corr, colors=[0, 200, 80], radii=0.0015),
    )

    # Draw ~500 randomly sampled correspondence lines in semi-transparent grey
    n_lines = min(500, len(src_corr))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(src_corr), size=n_lines, replace=False)
    strips = np.stack([src_corr[idx], dst_corr[idx]], axis=1)  # (N, 2, 3)

    rr.log(
        "pipeline_3d/correspondences/lines",
        rr.LineStrips3D(
            strips=strips,
            colors=[[0x88, 0x88, 0x88, 0x80]] * n_lines,
            radii=0.0005,
        ),
    )
    print(f"  [✓] Step 5 — correspondences  ({len(src_corr):,} total, {n_lines} drawn)")
    return src_corr, dst_corr


def log_umeyama_aligned(
        data: dict,
        src_corr: np.ndarray,
        dst_corr: np.ndarray,
        view_names: list[str],
        frame_idx: int,
        dataset_root: str,
        dataset_type: str,
) -> None:
    """Step 6 — Log aligned model (blue) and GT (green) point clouds."""
    rr.set_time("pipeline_step", sequence=6)

    s, R, tr = estimate_similarity_transform(src_corr, dst_corr)
    print(f"  [UMEYAMA] scale={s:.4f}  corr={len(src_corr):,}")

    V, H, W = normalize_spatial_dims(data)
    pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
    conf = (
        normalize_array(data["pointmaps_confs"], V, H, W)
        if "pointmaps_confs" in data
        else None
    )

    depth_max = DATASETS.get(dataset_type, {}).get("depth_max_m", DEPTH_MAX_M) or DEPTH_MAX_M
    validity_masks = build_gt_validity_masks(
        frame_idx, view_names, dataset_root,
        depth_max_m=depth_max,
        target_hw=(H, W),
        dataset_type=dataset_type,
    )

    frame_thr = float(data.get("min_conf_thr", 0.0))
    if frame_thr == 0.0 and conf is not None:
        frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE)

    est_parts: list[np.ndarray] = []
    for v in range(V):
        mask = np.ones((H, W), dtype=bool)
        if validity_masks[v] is not None:
            vmask = validity_masks[v]
            if vmask.shape != (H, W):
                vmask = cv2.resize(vmask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask &= vmask
        if conf is not None:
            mask &= conf[v] > frame_thr

        p = pm[v][mask]
        if len(p) > 0:
            est_parts.append(p)

    est_pts = np.concatenate(est_parts, axis=0) if est_parts else np.empty((0, 3))
    aligned_pts = apply_similarity_transform(est_pts, s, R, tr)

    gt_pts = build_gt_pointcloud(frame_idx, view_names, dataset_root, dataset_type=dataset_type)

    # Log aligned model points in blue — should overlap with GT after Umeyama
    rr.log(
        "pipeline_3d/aligned/model",
        rr.Points3D(positions=aligned_pts, colors=[0, 100, 255], radii=0.002),
    )

    if gt_pts is not None:
        # Log ground-truth points in green for comparison
        rr.log(
            "pipeline_3d/aligned/gt",
            rr.Points3D(positions=gt_pts, colors=[0, 200, 80], radii=0.002),
        )

    print("  [✓] Step 6 — umeyama_aligned")


# ═══════════════════════════════════════════════════════════════════════════════
#  4D Pipeline  (Steps 7 – 10)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_frame_points(
        path: str,
        dataset_root: str,
        dataset_type: str,
        apply_gt_mask: bool = True,
        apply_conf_filter: bool = True,
) -> tuple[np.ndarray, int, list[str]]:
    """Extract filtered 3D points + frame_idx from a single NPZ file."""
    data = np.load(path, allow_pickle=True)
    V, H, W = normalize_spatial_dims(data)
    pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
    conf = (
        normalize_array(data["pointmaps_confs"], V, H, W)
        if "pointmaps_confs" in data
        else None
    )
    t = int(data["frame_idx"])
    ks = data["Ks"]
    view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in ks]

    if apply_gt_mask:
        depth_max = DATASETS.get(dataset_type, {}).get("depth_max_m", DEPTH_MAX_M) or DEPTH_MAX_M
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=depth_max, target_hw=(H, W),
                                         dataset_type=dataset_type)
    else:
        vmasks = [None] * V

    frame_thr = 0.0
    if apply_conf_filter and conf is not None:
        frame_thr = float(data.get("min_conf_thr", 0.0))
        if frame_thr == 0.0:
            frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE)

    parts: list[np.ndarray] = []
    for v in range(V):
        mask = np.ones((H, W), dtype=bool)
        if vmasks[v] is not None:
            m = vmasks[v]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask &= m
        if conf is not None and apply_conf_filter:
            mask &= conf[v] > frame_thr
        p = pm[v][mask]
        if len(p) > 0:
            parts.append(p)

    pts = np.concatenate(parts, axis=0) if parts else np.empty((0, 3))
    return pts, t, view_names


def log_4d_unaligned_frames(
        frame_paths: list[str],
        dataset_root: str,
        dataset_type: str,
) -> None:
    """Step 7 — Log all frames in their arbitrary local coordinate spaces to show they are unaligned."""
    rr.set_time("pipeline_step", sequence=7)

    # Use a colour palette so each frame gets a distinct hue.
    n = len(frame_paths)
    cmap = matplotlib.colormaps["tab20"].resampled(n)

    for i, path in enumerate(frame_paths):
        pts, t, _ = _load_frame_points(path, dataset_root, dataset_type)
        if len(pts) == 0:
            continue

        color_rgba = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)

        # Log each frame as a separate entity in its raw local coordinate space
        rr.log(
            f"pipeline_4d/unaligned/frame_{i:03d}",
            rr.Points3D(positions=pts, colors=color_rgba.tolist(), radii=0.002),
        )

    print(f"  [✓] Step 7 — 4d_unaligned_frames  ({n} frames)")


def log_4d_flow_masks(
        frame_paths: list[str],
        dataset_root: str,
        dataset_type: str,
        target_frame: int,
) -> None:
    """Step 8 — Log the optical-flow dynamic mask and static point cloud for the selected frame."""
    rr.set_time("pipeline_step", sequence=8)

    for i, path in enumerate(frame_paths):
        data = np.load(path, allow_pickle=True)
        t = int(data["frame_idx"])
        if t != target_frame:
            continue

        V, H, W = normalize_spatial_dims(data)
        pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
        ks = data["Ks"]
        view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in ks]

        # The static / dynamic mask is stored in masks_2d
        if "masks_2d" not in data:
            print(f"  [WARN] Step 8 — frame {t}: no masks_2d, skipping")
            continue

        m_static = normalize_array(data["masks_2d"], V, H, W, is_mask=True)

        # Log flow masks as binary images: white (255) for static, black (0) for dynamic
        for v in range(V):
            mask_img = (m_static[v].astype(np.uint8) * 255)
            rr.log(
                f"pipeline_4d/flow_masks/frame_{i:03d}/masks/{view_names[v]}",
                rr.Image(mask_img),
            )

        # Log the static-only point cloud with its actual RGB colours
        rgbs = _extract_rgb_for_views(data, view_names, t, dataset_root, dataset_type)
        static_pts: list[np.ndarray] = []
        static_cols: list[np.ndarray] = []

        # Compute GT validity masks and confidence thresholds for cleaning the points
        conf = (
            normalize_array(data["pointmaps_confs"], V, H, W)
            if "pointmaps_confs" in data
            else None
        )
        depth_max = DATASETS.get(dataset_type, {}).get("depth_max_m", DEPTH_MAX_M) or DEPTH_MAX_M
        vmasks = build_gt_validity_masks(
            t, view_names, dataset_root, depth_max_m=depth_max, target_hw=(H, W), dataset_type=dataset_type
        )

        frame_thr = 0.0
        if conf is not None:
            frame_thr = float(data.get("min_conf_thr", 0.0))
            if frame_thr == 0.0:
                frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE)

        for v in range(V):
            mask = m_static[v]
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

            # Apply GT validity mask
            if vmasks[v] is not None:
                m = vmasks[v]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask &= m

            # Apply confidence filter
            if conf is not None:
                mask &= conf[v] > frame_thr

            p = pm[v][mask]
            if len(p) > 0:
                static_pts.append(p)
                if rgbs[v] is not None:
                    img = cv2.resize(rgbs[v], (W, H), interpolation=cv2.INTER_LINEAR)
                    c = img[mask]
                    static_cols.append(c)
                else:
                    static_cols.append(np.full((len(p), 3), 255, dtype=np.uint8))

        if static_pts:
            s_pts = np.concatenate(static_pts, axis=0)
            s_cols = np.concatenate(static_cols, axis=0)
            rr.log(
                f"pipeline_4d/flow_masks/frame_{i:03d}/static_pointcloud",
                rr.Points3D(positions=s_pts, colors=s_cols, radii=0.002),
            )

    print(f"  [✓] Step 8 — 4d_flow_masks  (frame {target_frame})")


def log_4d_concatenated_model(
        frame_paths: list[str],
        frame_transforms: list[tuple],
        s_glob: float,
        R_glob: np.ndarray,
        tr_glob: np.ndarray,
        dataset_root: str,
        dataset_type: str,
) -> None:
    """Step 9 — Animate the temporal accumulation of the aligned model."""
    accumulated: list[np.ndarray] = []
    accumulated_cols: list[np.ndarray] = []

    for i, path in enumerate(frame_paths):
        data = np.load(path, allow_pickle=True)
        V, H, W = normalize_spatial_dims(data)
        if H == 0:
            continue
        pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
        conf = (
            normalize_array(data["pointmaps_confs"], V, H, W)
            if "pointmaps_confs" in data
            else None
        )
        t = int(data["frame_idx"])
        ks = data["Ks"]
        view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in ks]
        depth_max = DATASETS.get(dataset_type, {}).get("depth_max_m", DEPTH_MAX_M) or DEPTH_MAX_M
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=depth_max, target_hw=(H, W),
                                         dataset_type=dataset_type)
        rgbs = _extract_rgb_for_views(data, view_names, t, dataset_root, dataset_type)

        s_i, R_i, tr_i = frame_transforms[i]
        s_tot = s_glob * s_i
        R_tot = R_glob @ R_i
        tr_tot = s_glob * (R_glob @ tr_i) + tr_glob

        frame_thr = 0.0
        if conf is not None:
            frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE)

        parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        for v in range(V):
            mask = np.ones((H, W), dtype=bool)
            if vmasks[v] is not None:
                m = vmasks[v]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask &= m
            if conf is not None:
                mask &= conf[v] > frame_thr
            p = pm[v][mask]
            if len(p) > 0:
                parts.append(apply_similarity_transform(p, s_tot, R_tot, tr_tot))
                if rgbs[v] is not None:
                    img = cv2.resize(rgbs[v], (W, H), interpolation=cv2.INTER_LINEAR)
                    c = img[mask]
                    col_parts.append(c)
                else:
                    col_parts.append(np.full((len(p), 3), 255, dtype=np.uint8))

        if parts:
            accumulated.append(np.concatenate(parts, axis=0))
            accumulated_cols.append(np.concatenate(col_parts, axis=0))

        # Animate accumulation: log the growing concatenated cloud at each time step.
        rr.set_time("pipeline_step", sequence=9)
        rr.set_time("4d_accumulation", sequence=i)
        merged = np.concatenate(accumulated, axis=0) if accumulated else np.empty((0, 3))
        merged_cols = np.concatenate(accumulated_cols, axis=0) if accumulated_cols else np.empty((0, 3), dtype=np.uint8)

        # Log accumulated model cloud — one step per frame to animate growth
        rr.log(
            "pipeline_4d/aligned_sequence/model",
            rr.Points3D(positions=merged, colors=merged_cols, radii=0.002),
        )

    print(f"  [✓] Step 9 — 4d_concatenated_model  ({len(frame_paths)} frames)")


def log_4d_final_alignment(
        frame_paths: list[str],
        frame_transforms: list[tuple],
        s_glob: float,
        R_glob: np.ndarray,
        tr_glob: np.ndarray,
        dataset_root: str,
        dataset_type: str,
) -> None:
    """Step 10 — Log the final globally-aligned model (blue) and GT (green)."""
    rr.set_time("pipeline_step", sequence=10)

    all_model: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []

    for i, path in enumerate(frame_paths):
        data = np.load(path, allow_pickle=True)
        V, H, W = normalize_spatial_dims(data)
        if H == 0:
            continue
        pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
        conf = (
            normalize_array(data["pointmaps_confs"], V, H, W)
            if "pointmaps_confs" in data
            else None
        )
        t = int(data["frame_idx"])
        ks = data["Ks"]
        view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in ks]
        depth_max = DATASETS.get(dataset_type, {}).get("depth_max_m", DEPTH_MAX_M) or DEPTH_MAX_M
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=depth_max, target_hw=(H, W),
                                         dataset_type=dataset_type)

        s_i, R_i, tr_i = frame_transforms[i]
        s_tot = s_glob * s_i
        R_tot = R_glob @ R_i
        tr_tot = s_glob * (R_glob @ tr_i) + tr_glob

        frame_thr = 0.0
        if conf is not None:
            frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE)

        parts: list[np.ndarray] = []
        for v in range(V):
            mask = np.ones((H, W), dtype=bool)
            if vmasks[v] is not None:
                m = vmasks[v]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask &= m
            if conf is not None:
                mask &= conf[v] > frame_thr
            p = pm[v][mask]
            if len(p) > 0:
                parts.append(apply_similarity_transform(p, s_tot, R_tot, tr_tot))

        if parts:
            all_model.append(np.concatenate(parts, axis=0))

        # GT
        gt_pts = build_gt_pointcloud(t, view_names, dataset_root, dataset_type=dataset_type)
        if gt_pts is not None:
            all_gt.append(gt_pts)

    model_merged = np.concatenate(all_model, axis=0) if all_model else np.empty((0, 3))
    gt_merged = np.concatenate(all_gt, axis=0) if all_gt else np.empty((0, 3))

    # Log final aligned model — all frames merged, coloured blue
    rr.log(
        "pipeline_4d/final/model",
        rr.Points3D(positions=model_merged, colors=[0, 100, 255], radii=0.002),
    )

    # Log final aligned GT — all frames merged, coloured green
    rr.log(
        "pipeline_4d/final/gt",
        rr.Points3D(positions=gt_merged, colors=[0, 200, 80], radii=0.002),
    )

    print(f"  [✓] Step 10 — 4d_final_alignment  (model={len(model_merged):,}  gt={len(gt_merged):,})")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_output_dir(
        output_root: str,
        dataset: str,
        strategy: str,
        data_dir: str,
        nviews: int,
) -> str:
    """Build the standard path: <output_root>/<dataset>/<strategy>/<sequence>/<nviews>views/

    The <sequence> component is inferred as the basename of *data_dir*.
    """
    sequence = os.path.basename(os.path.normpath(data_dir))
    # dataset folder uses a short name: dex-ycb, hi4d, …
    ds_folder = "dex-ycb" if dataset in ("dexycb", "dex-ycb") else dataset
    path = os.path.join(output_root, ds_folder, strategy, sequence, f"{nviews}views")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise the 3D / 4D reconstruction evaluation pipeline in Rerun."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset root for one subject/sequence.",
    )

    # ── Path resolution: use EITHER --model_output OR (--output_root + --strategy + --nviews) ──
    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument(
        "--model_output",
        type=str,
        default=None,
        help=(
            "Full path to the directory containing frame_*.npz files. "
            "Mutually exclusive with --output_root."
        ),
    )
    path_group.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Root of the aligned-outputs tree "
            "(e.g. aligned_outputs/pi3). "
            "The script resolves the full path as: "
            "<output_root>/<dataset>/<strategy>/<sequence>/<nviews>views/. "
            "Mutually exclusive with --model_output."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dexycb", "hi4d"],
        default="dexycb",
        help="Dataset type (default: dexycb).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="baseline",
        help=(
            "Alignment strategy sub-folder under <output_root>/<dataset>/ "
            "(e.g. baseline, strategy1, strategy2, strategy3). "
            "Only used when --output_root is set. Default: baseline."
        ),
    )
    parser.add_argument(
        "--nviews",
        type=int,
        default=4,
        help=(
            "Number-of-views folder (e.g. 2, 3, 4). "
            "Only used when --output_root is set. Default: 4."
        ),
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=14,
        help="Frame index to use for the 3D baseline (default: 14).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        metavar="STEPS",
        help=(
            "Comma-separated step numbers to run, or one of the shorthands: "
            "'all' (default), '3d' (steps 0-6), '4d' (steps 7-10). "
            "Examples: --steps 3d   --steps 4d   --steps 0,1,2   --steps 6,9,10"
        ),
    )
    args = parser.parse_args()

    # Normalise dataset_type to the internal convention used by the codebase.
    dataset_type: str = "dex-ycb" if args.dataset == "dexycb" else args.dataset
    data_dir: str = args.data_dir
    frame_idx: int = args.frame
    steps_to_run: set[int] = _parse_steps(args.steps)
    print(f"[INFO] Running steps: {sorted(steps_to_run)}")

    # ── Resolve the model-output directory ───────────────────────────────────
    if args.model_output is not None:
        model_output: str = args.model_output
    else:
        model_output = _resolve_output_dir(
            args.output_root, args.dataset, args.strategy, data_dir, args.nviews
        )
        print(f"[INFO] Resolved model output:  {model_output}")

    # ── Collect frames ────────────────────────────────────────────────────────
    frame_paths = _collect_frame_paths(model_output)
    if not frame_paths:
        print(f"[ERROR] No frame_*.npz files found in {model_output}")
        print(
            f"  → Expected layout:  "
            f"<output_root>/<dataset>/<strategy>/<sequence>/<nviews>views/frame_*.npz"
        )
        if args.output_root:
            print(f"  → Tried:  {model_output}")
            print(
                f"  → Check that --strategy ({args.strategy}), "
                f"--nviews ({args.nviews}), and --data_dir are all correct."
            )
        sys.exit(1)
    print(f"[INFO] Found {len(frame_paths)} frame files in {model_output}")

    # ── Select the single frame for the 3D pipeline ───────────────────────────
    frame_path_3d: Optional[str] = None
    for p in frame_paths:
        fname = os.path.basename(p)
        fid = int(fname.replace("frame_", "").replace(".npz", ""))
        if fid == frame_idx:
            frame_path_3d = p
            break

    if frame_path_3d is None:
        print(
            f"[WARN] Frame {frame_idx} not found in model output; "
            f"falling back to the first available frame."
        )
        frame_path_3d = frame_paths[0]
        frame_idx = int(
            os.path.basename(frame_path_3d)
            .replace("frame_", "")
            .replace(".npz", "")
        )
        print(f"[INFO] Using frame {frame_idx} instead.")

    # ── Rerun initialisation ──────────────────────────────────────────────────
    rr.init("eval_pipeline", spawn=True)

    # Configure default view orientation
    eye_up = DATASETS.get(dataset_type, {}).get("eye_up", RERUN_EYE_UP)
    print(f"[RERUN] Using eye_up={eye_up} for dataset={dataset_type}")

    # Always set the up-axis coordinate system
    for log_root in ["pipeline_3d", "pipeline_4d"]:
        rr.log(log_root, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Build a unified blueprint so we don't overwrite the 3D view with the 4D view
    try:
        import rerun.blueprint as rrb

        # Helper to try different Rerun SDK versions for EyeControls3D
        eye_controls = None
        if hasattr(rrb, "EyeControls3D"):
            eye_controls = rrb.EyeControls3D(eye_up=eye_up)
        elif hasattr(rrb, "archetypes") and hasattr(rrb.archetypes, "EyeControls3D"):
            eye_controls = rrb.archetypes.EyeControls3D(eye_up=eye_up)

        views = []
        if steps_to_run & set(range(7)):
            v3d = rrb.Spatial3DView(origin="pipeline_3d", name="3D Pipeline")
            if eye_controls:
                v3d = rrb.Spatial3DView(origin="pipeline_3d", name="3D Pipeline", eye_controls=eye_controls)
            elif hasattr(rrb, "Spatial3DView") and "eye_up" in rrb.Spatial3DView.__init__.__code__.co_varnames:
                v3d = rrb.Spatial3DView(origin="pipeline_3d", name="3D Pipeline", eye_up=eye_up)
            views.append(v3d)

        if steps_to_run & {7, 8, 9, 10}:
            v4d = rrb.Spatial3DView(origin="pipeline_4d", name="4D Pipeline")
            if eye_controls:
                v4d = rrb.Spatial3DView(origin="pipeline_4d", name="4D Pipeline", eye_controls=eye_controls)
            elif hasattr(rrb, "Spatial3DView") and "eye_up" in rrb.Spatial3DView.__init__.__code__.co_varnames:
                v4d = rrb.Spatial3DView(origin="pipeline_4d", name="4D Pipeline", eye_up=eye_up)
            views.append(v4d)

        if views:
            container = rrb.Horizontal(*views) if len(views) > 1 else views[0]
            rr.send_blueprint(rrb.Blueprint(container))
    except Exception as e:
        print(f"[WARN] Failed to configure blueprint: {e}")

    # ── Load frame data ───────────────────────────────────────────────────────
    data = _load_frame_npz(frame_path_3d)
    V, H, W = normalize_spatial_dims(data)
    view_names = _recover_view_names(data, data_dir, dataset_type)
    print(f"[INFO] 3D pipeline: frame={frame_idx}, views={view_names}, resolution={H}×{W}")

    has_confidence = "pointmaps_confs" in data

    # ══════════════════════════════════════════════════════════════════════════
    #  3D Pipeline  (Steps 0 – 6)
    # ══════════════════════════════════════════════════════════════════════════

    run_3d = steps_to_run & set(range(7))
    if run_3d:
        print("\n─── 3D Pipeline ────────────────────────────────────────────────")

        # Steps 2-6 all depend on validity_masks; steps 3-6 on masked_pts.
        # Compute them lazily only when a downstream step is requested.
        validity_masks = None
        masked_pts = masked_colors = masked_confs = None
        src_corr = dst_corr = None

        if 0 in steps_to_run:
            log_raw_pointcloud(data, view_names, frame_idx, data_dir, dataset_type)

        if steps_to_run & {1, 2, 3, 4, 5, 6}:
            validity_masks = log_gt_validity_mask(
                data, view_names, frame_idx, data_dir, dataset_type
            ) if 1 in steps_to_run else [
                # build masks silently for downstream steps
                *[m for m in __import__('pi3.utils.gt', fromlist=['build_gt_validity_masks'])
                .build_gt_validity_masks(
                    frame_idx, view_names, data_dir,
                    depth_max_m=DATASETS.get(dataset_type, {}).get('depth_max_m', DEPTH_MAX_M) or DEPTH_MAX_M,
                    target_hw=(normalize_spatial_dims(data)[1], normalize_spatial_dims(data)[2]),
                    dataset_type=dataset_type,
                )]
            ]

        if steps_to_run & {2, 3, 4, 5, 6}:
            masked_pts, masked_colors, masked_confs = log_masked_pointcloud(
                data, validity_masks, view_names, frame_idx, data_dir, dataset_type, do_log=(2 in steps_to_run)
            )

        if 3 in steps_to_run:
            if has_confidence:
                log_confidence_coloring(masked_pts, masked_confs)
            else:
                print("  [WARN] Step 3 — skipped (no confidence map available)")

        if 4 in steps_to_run:
            if has_confidence:
                log_confidence_filtered(masked_pts, masked_colors, masked_confs)
            else:
                print("  [WARN] Step 4 — skipped (no confidence map available)")

        if steps_to_run & {5, 6}:
            src_corr, dst_corr = log_correspondences(
                data, view_names, frame_idx, data_dir, dataset_type
            ) if 5 in steps_to_run else get_static_correspondences(
                frame_idx, view_names,
                [normalize_array(data["pointmaps"], *normalize_spatial_dims(data))[v]
                 for v in range(normalize_spatial_dims(data)[0])],
                [normalize_array(data["pointmaps_confs"], *normalize_spatial_dims(data))[v]
                 if "pointmaps_confs" in data
                 else np.ones(normalize_spatial_dims(data)[1:])
                 for v in range(normalize_spatial_dims(data)[0])],
                data_dir,
                conf_percentile=CONF_PERCENTILE,
                use_static_mask=False,
                dataset_type=dataset_type,
            )

        if 6 in steps_to_run:
            if src_corr is not None and dst_corr is not None and len(src_corr) >= 3:
                log_umeyama_aligned(
                    data, src_corr, dst_corr, view_names, frame_idx, data_dir, dataset_type
                )
            else:
                print("  [WARN] Step 6 — skipped (insufficient correspondences)")

    # ══════════════════════════════════════════════════════════════════════════
    #  4D Pipeline  (Steps 7 – 10)
    # ══════════════════════════════════════════════════════════════════════════

    run_4d = steps_to_run & {7, 8, 9, 10}
    if run_4d:
        if len(frame_paths) < 2:
            print("\n[INFO] Only 1 frame available — skipping 4D pipeline.")
        else:
            print(f"\n─── 4D Pipeline (Strategy 2, {len(frame_paths)} frames) ──────────────────────")

            if 7 in steps_to_run:
                log_4d_unaligned_frames(frame_paths, data_dir, dataset_type)

            if 8 in steps_to_run:
                log_4d_flow_masks(frame_paths, data_dir, dataset_type, frame_idx)

            if steps_to_run & {9, 10}:
                # Alignment computation is needed for steps 9 and/or 10.
                print("  [4D] Running strategy-2 hierarchical alignment …")
                frame_transforms = strategy2_hierarchical(
                    frame_paths, data_dir, dataset_type=dataset_type
                )
                print("  [4D] Solving global GT registration …")
                s_glob, R_glob, tr_glob = solve_final_gt_registration(
                    frame_paths, frame_transforms, data_dir,
                    use_static_mask=False, dataset_type=dataset_type,
                )

                if 9 in steps_to_run:
                    log_4d_concatenated_model(
                        frame_paths, frame_transforms, s_glob, R_glob, tr_glob,
                        data_dir, dataset_type,
                    )

                if 10 in steps_to_run:
                    log_4d_final_alignment(
                        frame_paths, frame_transforms, s_glob, R_glob, tr_glob,
                        data_dir, dataset_type,
                    )

    # ══════════════════════════════════════════════════════════════════════════
    #  Wrap-up
    # ══════════════════════════════════════════════════════════════════════════

    os.makedirs("figures", exist_ok=True)

    print("\n" + "═" * 70)
    print("  DONE — All pipeline steps have been logged to Rerun.")
    print("═" * 70)
    print(
        "\n📸 Screenshot checklist — take a Rerun screenshot for each step"
        "\n   and save it into figures/.\n"
    )
    checklist = [
        ("Step  0", "pipeline_step=0", "pipeline_3d/raw_pointcloud"),
        ("Step  1", "pipeline_step=1", "pipeline_3d/gt_depth_image  +  pipeline_3d/gt_validity_mask"),
        ("Step  2", "pipeline_step=2", "pipeline_3d/masked_pointcloud"),
        ("Step  3", "pipeline_step=3", "pipeline_3d/confidence_colored"),
        ("Step  4", "pipeline_step=4", "pipeline_3d/confidence_filtered"),
        ("Step  5", "pipeline_step=5", "pipeline_3d/correspondences/{estimated,gt,lines}"),
        ("Step  6", "pipeline_step=6", "pipeline_3d/aligned/{model,gt}"),
        ("Step  7", "pipeline_step=7", "pipeline_4d/unaligned/frame_*"),
        ("Step  8", "pipeline_step=8", "pipeline_4d/flow_masks/frame_*/masks/* + static_pointcloud"),
        ("Step  9", "pipeline_step=9", "pipeline_4d/aligned_sequence/model  (scrub 4d_accumulation)"),
        ("Step 10", "pipeline_step=10", "pipeline_4d/final/{model,gt}"),
    ]
    for step, timeline, entities in checklist:
        print(f"  {step}  [{timeline}]  →  {entities}")

    print(f"\n  Save screenshots to:  {os.path.abspath('figures')}/")
    print()

    # Keep the script alive so Rerun has time to receive data and the user can capture screenshots
    print("  [RERUN] The viewer is now active.")
    try:
        input("  Press Enter here in the terminal when you are done capturing screenshots to exit... ")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
