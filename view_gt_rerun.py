#!/usr/bin/env python3
"""
view_gt_rerun.py  –  Ground Truth visualization for DEX-YCB multi-view sequence
================================================================================

This script loads the ground truth (GT) depth and camera parameters from the
DEX-YCB dataset and visualizes them in Rerun.

SSH tunnel setup:
    RemoteForward 9876 localhost:9876

Workflow:
1. Launch `rerun` on your Mac (it auto-listens on 0.0.0.0:9876).
2. SSH into the remote: `ssh vlg`
3. Run this script: `python view_gt_rerun.py`
"""

import os
import glob
import numpy as np
import cv2
import rerun as rr
from PIL import Image

# ── configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"
RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"
DEPTH_SCALE = 1000.0  # Assuming depth is in mm, convert to meters if needed. Adjust as necessary.


# ── helpers ───────────────────────────────────────────────────────────────────

def load_gt_cameras(view_dir):
    npz_path = os.path.join(view_dir, "intrinsics_extrinsics.npz")

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        print(data.files) # Checking for cam2world or world2cam

        intrinsics = None
        extrinsics = None

        if 'intrinsics' in data:
            intrinsics = data['intrinsics']
        elif 'K' in data:
            intrinsics = data['K']

        if 'extrinsics' in data:
            extrinsics = data['extrinsics']
        elif 'cam2world' in data:
            extrinsics = data['cam2world']

        if extrinsics is not None:
            extrinsics = np.linalg.inv(extrinsics)

        return intrinsics, extrinsics

    txt_path = os.path.join(view_dir, "intrinsics.txt")
    if os.path.exists(txt_path):
        intrinsics = np.loadtxt(txt_path)
        return intrinsics, None

    return None, None


def project_depth_to_3d(depth, intrinsics, extrinsics):
    """Unproject depth map to 3D points in world coordinates."""
    h, w = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Create coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Mask valid depth
    mask = depth > 0
    u = u[mask]
    v = v[mask]
    z = depth[mask] / DEPTH_SCALE

    # Unproject to camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    # Transform to world coordinates
    if extrinsics is not None:
        # P_world = R * P_cam + t
        # extrinsics is usually 4x4 cam2world
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        pts_world = (R @ pts_cam.T).T + t
        return pts_world
    else:
        return pts_cam


def build_view_data(dataset_root):
    """Collect view directories and frame counts."""
    view_dirs = sorted(glob.glob(os.path.join(dataset_root, "view_*")))
    views = {}
    for vd in view_dirs:
        vname = os.path.basename(vd)
        depth_dir = os.path.join(vd, "depth")
        rgb_dir = os.path.join(vd, "rgb")

        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

        if depth_files:
            views[vname] = {
                'dir': vd,
                'depth_files': depth_files,
                'rgb_files': rgb_files if len(rgb_files) == len(depth_files) else None
            }
    return views


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[gt] scanning dataset at {DATASET_ROOT} ...")
    views = build_view_data(DATASET_ROOT)
    if not views:
        print("[error] no views found!")
        return

    view_names = sorted(views.keys())
    n_frames = len(views[view_names[0]]['depth_files'])
    print(f"[gt] found {len(view_names)} views x {n_frames} frames")

    rr.init("mast3r_gt_viz", spawn=False)
    rr.connect_grpc(RERUN_ADDR)
    print(f"[gt] streaming to {RERUN_ADDR}")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # -------------------------
    # Load cameras ONCE
    # -------------------------
    camera_data = {}

    for vname in view_names:
        intrinsics, extrinsics = load_gt_cameras(views[vname]['dir'])
        camera_data[vname] = (intrinsics, extrinsics)

    # -------------------------
    # Frame loop
    # -------------------------
    for t in range(n_frames):

        print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")
        rr.set_time("timestep", sequence=t)

        all_gt_pts = []
        all_gt_cols = []

        for vname in view_names:

            vdata = views[vname]
            intrinsics, extrinsics = camera_data[vname]

            if intrinsics is None:
                continue

            depth_path = vdata['depth_files'][t]
            depth = np.array(Image.open(depth_path))

            rgb = None
            if vdata['rgb_files']:
                rgb_path = vdata['rgb_files'][t]
                rgb = np.array(Image.open(rgb_path))

            entity_path = f"world/gt/cameras/{vname}"

            if extrinsics is not None:
                rr.log(entity_path, rr.Pinhole(
                    focal_length=intrinsics[0, 0],
                    width=depth.shape[1],
                    height=depth.shape[0],
                ))

                rr.log(entity_path, rr.Transform3D(
                    translation=extrinsics[:3, 3],
                    mat3x3=extrinsics[:3, :3],
                ))

                if rgb is not None:
                    rr.log(f"{entity_path}/rgb", rr.Image(rgb))

            pts_world = project_depth_to_3d(depth, intrinsics, extrinsics)
            all_gt_pts.append(pts_world)

            if rgb is not None:
                mask = depth > 0
                cols = rgb[mask]
                all_gt_cols.append(cols)

        if all_gt_pts:
            pts_cat = np.concatenate(all_gt_pts, axis=0)[::5]

            if all_gt_cols:
                cols_cat = np.concatenate(all_gt_cols, axis=0)[::5]

                rr.log("world/gt/point_cloud", rr.Points3D(
                    positions=pts_cat,
                    colors=cols_cat,
                    radii=0.002,
                ))
            else:
                rr.log("world/gt/point_cloud", rr.Points3D(
                    positions=pts_cat,
                    radii=0.002,
                ))

            print(f"  ✓ t={t:02d} logged {len(pts_cat)} points")

    print("[done] ground truth visualization finished.")
if __name__ == "__main__":
    main()
