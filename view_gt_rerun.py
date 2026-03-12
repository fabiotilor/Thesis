#!/usr/bin/env python3
import os
import glob
import numpy as np
import cv2
import torch
import rerun as rr

# ── configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"
IMAGE_SIZE = 512
RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"
DEPTH_SCALE = 0.001        # Convert mm → metres
DEPTH_MAX_M = 2.0          # Discard depths beyond 2 m (uint16 max = sentinel)
SUBSAMPLE   = 1


# ── helpers ───────────────────────────────────────────────────────────────────

def build_views(dataset_root: str) -> dict:
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
    """
    Returns (K, cam2world) both as float64.
    'extrinsics' in DEX-YCB npz is world2cam -> invert to get cam2world.
    'intrinsics' is stored as 4x4 -> take top-left 3x3.
    """
    data = np.load(os.path.join(view_dir, "intrinsics_extrinsics.npz"))

    K_raw = data['intrinsics'].astype(np.float64)
    K = K_raw[:3, :3]   # top-left 3x3 (handles both 3x3 and 4x4 storage)

    world2cam = data['extrinsics'].astype(np.float64)
    cam2world = np.linalg.inv(world2cam)   # flip convention

    return K, cam2world


def backproject(depth_m: np.ndarray, K: np.ndarray):
    """
    Unproject a depth image (in metres) to camera-space points.
    Returns (pts_cam [Nx3], mask [HxW bool]).
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = (depth_m > 0) & (depth_m < DEPTH_MAX_M)
    u, v, z = u[mask], v[mask], depth_m[mask]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1), mask


def log_gt_timestep(t: int, view_names: list, views_dict: dict, dataset_root: str) -> None:
    # FIX 1: correct API - rr.set_time_sequence does not exist
    rr.set_time("timestep", sequence=t)

    all_pts  = []
    all_cols = []

    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        rgb_path = views_dict[vname][t]
        img_rgb  = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        H, W     = img_rgb.shape[:2]

        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        depth_raw  = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m    = depth_raw * DEPTH_SCALE   # mm -> metres

        # FIX 2: invert extrinsics (world2cam -> cam2world)
        K, cam2world = load_gt_params(view_dir)
        entity = f"world/cameras/{vname}"

        # intrinsics
        rr.log(entity, rr.Pinhole(
            image_from_camera    = K,
            width                = W,
            height               = H,
            image_plane_distance = 0.2,
        ))

        # pose (cam2world: translation = camera origin in world)
        rr.log(entity, rr.Transform3D(
            translation = cam2world[:3, 3],
            mat3x3      = cam2world[:3, :3],
        ))

        rr.log(f"{entity}/rgb", rr.Image(img_rgb))

        # backproject + transform to world
        pts_cam, mask = backproject(depth_m, K)
        pts_world     = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
        cols          = img_rgb[mask]

        all_pts.append(pts_world[::SUBSAMPLE])
        all_cols.append(cols[::SUBSAMPLE])

    if all_pts:
        rr.log("world/point_cloud", rr.Points3D(
            positions = np.concatenate(all_pts,  axis=0),
            colors    = np.concatenate(all_cols, axis=0),
            radii     = 0.003,
        ))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    rr.init("mast3r_dexycb", spawn=False)
    rr.connect_grpc(RERUN_ADDR)
    print(f"[rerun] streaming GT to {RERUN_ADDR} (gRPC)")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    views_dict = build_views(DATASET_ROOT)
    view_names = sorted(views_dict.keys())
    n_frames   = len(views_dict[view_names[0]])
    print(f"[INFO] {len(view_names)} views x {n_frames} frames")

    for t in range(n_frames):
        print(f"-- t={t:02d} / {n_frames - 1} --")
        log_gt_timestep(t, view_names, views_dict, DATASET_ROOT)
        print(f"  GT t={t:02d} logged\n")

    print("[done] all GT timesteps streamed to Rerun.")


if __name__ == "__main__":
    main()