#!/usr/bin/env python3
"""
run_rerun.py  –  MASt3R multi-view reconstruction with live Rerun visualisation
================================================================================

SSH tunnel setup (already in your ~/.ssh/config):
    RemoteForward 9876 localhost:9876

Workflow
--------
1. On your **Mac**, open the Rerun viewer and listen for incoming connections:

Launch `rerun` and it will auto-listen on 0.0.0.0:9876.

2. SSH into the remote as usual:  `ssh vlg`

3. On the **remote**, run:

       cd /home/fabio/mast3r
       python run_rerun.py

Results (RGB images, camera frustums, coloured point clouds) appear in the viewer
on your Mac as each timestep finishes.

Configuration
-------------
Edit the CAPS constants below to change model / dataset / optimisation settings.
"""

import argparse
import os
import glob
import tempfile
import numpy as np
import torch
import cv2

# ── project path setup ───────────────────────────────────────────────────────
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

import rerun as rr

# ── configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"  # First subject of the DEX-YCB multi-view dataset augmented by MVTRACKER
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rerun TCP address – tunnelled to your Mac via RemoteForward 9876 localhost:9876
RERUN_ADDR = "127.0.0.1:9876"

# Reconstruction hyper-parameters
LR1, NITER1 = 0.07, 300
LR2, NITER2 = 0.01, 300
CONF_PERCENTILE = 0.5  # Filter to retain the top 50% of points based on confidence
SCENEGRAPH = "complete"  # "complete" | "swin" | "logwin" | "oneref"
CLEAN_DEPTH = True
OPT_DEPTH = True  # refine+depth mode
SHARED_INTRIN = False


# ── helpers ───────────────────────────────────────────────────────────────────

def build_views(dataset_root: str) -> dict:
    """Return {view_name: [sorted frame paths]} for the view_*/rgb layout."""
    from collections import defaultdict
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views: dict = defaultdict(list)
    view_dirs = sorted(glob.glob(os.path.join(dataset_root, "view_*")))
    for vd in view_dirs:
        vname = os.path.basename(vd)
        rgb_dir = os.path.join(vd, "rgb")
        search = rgb_dir if os.path.isdir(rgb_dir) else vd
        frames = sorted(
            f for f in glob.glob(os.path.join(search, "*"))
            if os.path.splitext(f.lower())[1] in img_exts
        )
        if frames:
            views[vname] = frames
    return dict(views)


def log_timestep(t: int, view_names: list, scene) -> None:
    """Log all scene data for one timestep to Rerun."""
    rr.set_time_sequence("timestep", t)

    rgbimgs = scene.imgs  # list[H×W×3 float32, 0..1]
    focals = to_numpy(scene.get_focals())  # (N,)
    cam2world = to_numpy(scene.get_im_poses())  # (N, 4, 4)

    # Dense point cloud + per-pixel confidence
    pts3d_list, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH))
    conf_masks = []
    for c in confs:
        thr = np.percentile(c, 100 * (1 - CONF_PERCENTILE))
        conf_masks.append(c > thr)

    all_pts = []
    all_cols = []

    for i, v in enumerate(view_names):
        img_f32 = np.array(rgbimgs[i], dtype=np.float32)  # H×W×3
        H, W = img_f32.shape[:2]
        img_u8 = (np.clip(img_f32, 0.0, 1.0) * 255).astype(np.uint8)

        focal_i = float(focals[i])
        c2w = cam2world[i]  # 4×4
        entity = f"world/cameras/{v}"

        # ── camera intrinsics ──────────────────────────────────────────
        rr.log(entity, rr.Pinhole(
            focal_length=focal_i,
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
        rr.log(f"{entity}/rgb", rr.Image(img_u8))

        # ── per-view points (masked by confidence + finite check) ──────
        pts_i = pts3d_list[i].reshape(-1, 3)
        msk = conf_masks[i].ravel() & np.isfinite(pts_i.sum(axis=1))
        all_pts.append(pts_i[msk])
        all_cols.append(img_u8.reshape(-1, 3)[msk])

    # ── fused colour point cloud ───────────────────────────────────────────
    if all_pts:
        pts_cat = np.concatenate(all_pts, axis=0)
        cols_cat = np.concatenate(all_cols, axis=0)
        rr.log("world/point_cloud", rr.Points3D(
            positions=pts_cat,
            colors=cols_cat,
            radii=0.003,
        ))


def get_masked_image(t: int, vname: str, rgb_path: str, mask_mode: str, cache_dir: str, dataset_root: str):
    if mask_mode == "none":
        return rgb_path

    view_dir = os.path.join(dataset_root, vname)
    # the dataset stores masks as e.g. 00000.png or 000000.png, assuming f"{t:05d}.png" based on depth format
    # wait, could they be 6 digits? Let's check view_gt_rerun.py. It uses f"{t:05d}.png".
    mask_path = os.path.join(view_dir, "mask", f"{t:05d}.png")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(view_dir, "mask", f"{t:06d}.png")

    img = cv2.imread(rgb_path)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if mask_mode == "masked":
            valid = mask > 0
        else:
            valid = mask == 0
        img[~valid] = 0

    out_name = f"{vname}_{t:05d}_masked.jpg"
    out_path = os.path.join(cache_dir, out_name)
    cv2.imwrite(out_path, img)
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
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
    print(f"[rerun] streaming to {RERUN_ADDR}  (tunnelled via SSH RemoteForward)")

    # world coordinate-system annotation (OpenGL: Y-up, right-handed)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # ── dataset ────────────────────────────────────────────────────────────
    views = build_views(DATASET_ROOT)
    view_names = sorted(views.keys())
    n_frames = len(views[view_names[0]])
    print(f"[INFO] {len(view_names)} views × {n_frames} frames  "
          f"→  {n_frames} independent reconstructions of {len(view_names)} images each")

    # ── model ──────────────────────────────────────────────────────────────
    print(f"[INFO] loading model '{MODEL_NAME}' on {DEVICE} …")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    print("[INFO] model ready\n")

    # ── camera stabilization ───────────────────────────────────────────────
    NUM_POSE_INIT_FRAMES = 5
    cached_camera_params = None  # {view_name: {'intrinsics': K, 'cam2w': pose}}

    # ── per-timestep loop ──────────────────────────────────────────────────
    cache_root = os.path.join(tempfile.gettempdir(), "mast3r_rerun_cache")
    os.makedirs(cache_root, exist_ok=True)

    for t in range(n_frames):
        print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")
        current_files = [views[v][t] for v in view_names]

        # Stage 1: Camera Estimation Phase (once)
        if cached_camera_params is None and t >= NUM_POSE_INIT_FRAMES - 1:
            print(f"[INFO] Stage 1: Estimating stable camera parameters from first {NUM_POSE_INIT_FRAMES} frames...")
            # We collect ALL images from ALL calibration frames
            calib_files = []
            for t_cal in range(NUM_POSE_INIT_FRAMES):
                calib_files.extend([
                    get_masked_image(t_cal, v, views[v][t_cal], args.mask_mode, cache_root, DATASET_ROOT)
                    for v in view_names
                ])

            calib_imgs = load_images(calib_files, size=IMAGE_SIZE, verbose=True)
            # Use a slightly more connected scene graph for calibration if needed,
            # but "complete" might be too heavy for N*8 images.
            # For 5 frames * 8 cameras = 40 images. "complete" is 40*39/2 = 780 pairs.
            # Let's stick to the default or a "swin" approach for calibration if N is large.
            calib_pairs = make_pairs(calib_imgs, scene_graph=SCENEGRAPH, prefilter=None, symmetrize=True)

            calib_cache = os.path.join(cache_root, "calibration")
            os.makedirs(calib_cache, exist_ok=True)

            calib_scene = sparse_global_alignment(
                calib_files, calib_pairs, calib_cache,
                model, lr1=LR1, niter1=NITER1, lr2=LR2, niter2=NITER2,
                device=DEVICE, opt_depth=OPT_DEPTH, shared_intrinsics=SHARED_INTRIN,
                matching_conf_thr=0.0
            )

            # Extract and average camera parameters per view
            # Note: sparse_global_alignment returns one pose per image.
            # We have NUM_POSE_INIT_FRAMES * num_views images.
            all_intrinsics = to_numpy(calib_scene.intrinsics)  # (N*V, 3, 3)
            all_poses = to_numpy(calib_scene.cam2w)  # (N*V, 4, 4)

            cached_camera_params = {}
            for i, vname in enumerate(view_names):
                # Indices for this view across all calibration frames: i, i+V, i+2V, ...
                indices = [i + (f * len(view_names)) for f in range(NUM_POSE_INIT_FRAMES)]

                # Average Intrinsics
                K_avg = np.mean(all_intrinsics[indices], axis=0)

                # Average Posets (properly averaging rotations via SVD or just taking the first one if small motion)
                # Since we assuming static cameras, just take the first one or average them.
                # Let's take the one from the middle frame for better stability.
                mid_idx = indices[NUM_POSE_INIT_FRAMES // 2]
                pose_ref = all_poses[mid_idx]

                cached_camera_params[vname] = {
                    'intrinsics': torch.from_numpy(K_avg).to(DEVICE),
                    'cam2w': torch.from_numpy(pose_ref).to(DEVICE)
                }
            print("[INFO] Camera parameters cached.\n")

        # Stage 2: Fixed Camera Reconstruction
        init_params = {}
        masked_current_files = [
            get_masked_image(t, v, views[v][t], args.mask_mode, cache_root, DATASET_ROOT)
            for v in view_names
        ]
        if cached_camera_params is not None:
            for i, v in enumerate(view_names):
                init_params[masked_current_files[i]] = {
                    'intrinsics': cached_camera_params[v]['intrinsics'],
                    'cam2w': cached_camera_params[v]['cam2w'],
                    'freeze_pose': True,
                    'freeze_intrinsics': True
                }

        imgs = load_images(masked_current_files, size=IMAGE_SIZE, verbose=True)
        pairs = make_pairs(imgs, scene_graph=SCENEGRAPH,
                           prefilter=None, symmetrize=True)

        cache_dir = os.path.join(cache_root, f"t{t:02d}")
        os.makedirs(cache_dir, exist_ok=True)

        scene = sparse_global_alignment(
            masked_current_files, pairs, cache_dir,
            model,
            lr1=LR1, niter1=NITER1,
            lr2=LR2, niter2=NITER2,
            device=DEVICE,
            opt_depth=OPT_DEPTH,
            shared_intrinsics=SHARED_INTRIN,
            matching_conf_thr=0.0,
            init=init_params  # Inject fixed parameters
        )

        log_timestep(t, view_names, scene)
        print(f"  ✓ t={t:02d} logged to Rerun\n")

    print("[done] all timesteps streamed to Rerun.")


if __name__ == "__main__":
    main()
