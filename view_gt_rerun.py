#!/usr/bin/env python3
"""
view_gt_rerun.py  –  Ground truth visualization with Rerun
==========================================================

This script loads ground truth RGB, depth, and camera parameters from the
DEX-YCB dataset and visualizes them in Rerun, maintaining the same structure
as run_rerun.py for easy comparison.
"""

import os
import glob
import numpy as np
import cv2
import torch
import mast3r.utils.path_to_dust3r  # noqa
import rerun as rr

# ── configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"
IMAGE_SIZE = 512
RERUN_ADDR = "127.0.0.1:9876"
DEPTH_SCALE = 0.001  # Convert mm to meters
NUM_FRAMES_TO_LOG = None  # Process all frames


# ── helpers ───────────────────────────────────────────────────────────────────

def build_views(dataset_root: str) -> dict:
    """Return {view_name: [sorted frame paths]} for the view_*/rgb layout."""
    from collections import defaultdict
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = defaultdict(list)
    view_dirs = sorted(glob.glob(os.path.join(dataset_root, "view_*")))
    for vd in view_dirs:
        vname = os.path.basename(vd)
        rgb_dir = os.path.join(vd, "rgb")
        if os.path.isdir(rgb_dir):
            frames = sorted(
                f for f in glob.glob(os.path.join(rgb_dir, "*"))
                if os.path.splitext(f.lower())[1] in img_exts
            )
            if frames:
                views[vname] = frames
    return dict(views)


def load_gt_params(view_dir: str):
    """Load intrinsics and extrinsics for a view."""
    path = os.path.join(view_dir, "intrinsics_extrinsics.npz")
    data = np.load(path)
    return data['intrinsics'], data['extrinsics']


def backproject(depth, intrinsics):
    """Backproject depth map to 3D points in camera frame."""
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Filter 0 depth
    mask = depth > 0
    u = u[mask]
    v = v[mask]
    z = depth[mask]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = np.stack([x, y, z], axis=-1)
    return pts_cam, mask


def log_gt_timestep(t: int, view_names: list, views_dict: dict, dataset_root: str, mask_mode: str) -> None:
    """Log ground truth data for one timestep to Rerun."""
    rr.set_time_sequence("timestep", t)

    all_pts = []
    all_cols = []

    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        rgb_path = views_dict[vname][t]

        # Load RGB
        img_bgr = cv2.imread(rgb_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # Load Depth
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * DEPTH_SCALE

        # Load Params
        K, c2w = load_gt_params(view_dir)

        entity = f"world/cameras/{vname}"

        # ── camera intrinsics ──────────────────────────────────────────
        rr.log(entity, rr.Pinhole(
            focal_length=K[0, 0],
            width=W,
            height=H,
            image_plane_distance=0.2,
        ))

        # ── camera extrinsics (cam-to-world) ───────────────────────────
        rr.log(entity, rr.Transform3D(
            translation=c2w[:3, 3],
            mat3x3=c2w[:3, :3],
        ))

        # ── RGB image inside the frustum ───────────────────────────────
        rr.log(f"{entity}/rgb", rr.Image(img_rgb))

        # ── points ─────────────────────────────────────────────────────
        pts_cam, orig_mask = backproject(depth, K)

        # Apply segmentation mask
        if mask_mode != "none":
            # load segmentation mask
            mask_path = os.path.join(view_dir, "mask", f"{t:05d}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(view_dir, "mask", f"{t:06d}.png")

            if os.path.exists(mask_path):
                seg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if seg_mask.shape[:2] != depth.shape[:2]:
                    seg_mask = cv2.resize(seg_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Transform to world
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        if mask_mode != "none" and os.path.exists(mask_path):
            if mask_mode == "masked":
                valid_seg = (seg_mask > 0)[orig_mask]
            else:
                valid_seg = (seg_mask == 0)[orig_mask]
            pts_world = pts_world[valid_seg]
            cols = img_rgb[orig_mask][valid_seg]
        else:
            cols = img_rgb[orig_mask]

        # Downsample for performance if needed
        step = 4
        all_pts.append(pts_world[::step])
        all_cols.append(cols[::step])

    # ── fused colour point cloud ───────────────────────────────────────────
    if all_pts:
        pts_cat = np.concatenate(all_pts, axis=0)
        cols_cat = np.concatenate(all_cols, axis=0)
        rr.log("world/point_cloud", rr.Points3D(
            positions=pts_cat,
            colors=cols_cat,
            radii=0.003,
        ))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_mode", type=str, choices=["none", "masked", "inverse_masked"], default="none")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+

    # ── connect to Rerun viewer ────────────────────────────────────────────
    rr.init("mast3r_dexycb", spawn=False)
    try:
        rr.connect_tcp(RERUN_ADDR)
    except AttributeError:
        # older rerun SDK (<0.14) uses rr.connect()
        rr.connect(RERUN_ADDR)
    print(f"[rerun] streaming GT to {RERUN_ADDR}")

    # world coordinate-system annotation (OpenGL: Y-up, right-handed)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # ── dataset ────────────────────────────────────────────────────────────
    views_dict = build_views(DATASET_ROOT)
    view_names = sorted(views_dict.keys())
    n_frames = len(views_dict[view_names[0]])
    print(f"[INFO] {len(view_names)} views × {n_frames} frames")

    # ── loop ───────────────────────────────────────────────────────────────
    stop_frame = min(n_frames, NUM_FRAMES_TO_LOG) if NUM_FRAMES_TO_LOG else n_frames
    for t in range(stop_frame):
        print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")
        log_gt_timestep(t, view_names, views_dict, DATASET_ROOT, args.mask_mode)
        print(f"  ✓ GT t={t:02d} logged to Rerun\n")

    print("[done] all GT timesteps streamed to Rerun.")


if __name__ == "__main__":
    main()
