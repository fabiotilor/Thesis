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

import os
import glob
import tempfile
import numpy as np
import torch

# ── project path setup ───────────────────────────────────────────────────────
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

import rerun as rr

# ── configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT  = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"
MODEL_NAME    = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
IMAGE_SIZE    = 512
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# FIX 1: correct address format (matches view_gt_rerun.py)
RERUN_ADDR    = "rerun+http://127.0.0.1:9876/proxy"

# Reconstruction hyper-parameters
LR1, NITER1   = 0.07, 300
LR2, NITER2   = 0.01, 300
MIN_CONF_THR  = 1.5
SCENEGRAPH    = "complete"
CLEAN_DEPTH   = True
OPT_DEPTH     = True
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
    # FIX 2: correct time API
    rr.set_time("timestep", sequence=t)

    rgbimgs   = scene.imgs
    focals    = to_numpy(scene.get_focals())
    cam2world = to_numpy(scene.get_im_poses())

    pts3d_list, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH))
    conf_masks = [c > MIN_CONF_THR for c in confs]

    all_pts  = []
    all_cols = []

    for i, v in enumerate(view_names):
        img_f32 = np.array(rgbimgs[i], dtype=np.float32)
        H, W    = img_f32.shape[:2]
        img_u8  = (np.clip(img_f32, 0.0, 1.0) * 255).astype(np.uint8)

        focal_i = float(focals[i])
        c2w     = cam2world[i]
        entity  = f"world/cameras/{v}"

        rr.log(entity, rr.Pinhole(
            focal_length=focal_i,
            width=W,
            height=H,
            image_plane_distance=0.2,
        ))
        rr.log(entity, rr.Transform3D(
            translation=c2w[:3, 3],
            mat3x3=c2w[:3, :3],
        ))
        rr.log(f"{entity}/rgb", rr.Image(img_u8))

        pts_i = pts3d_list[i].reshape(-1, 3)
        msk   = conf_masks[i].ravel() & np.isfinite(pts_i.sum(axis=1))
        all_pts.append(pts_i[msk])
        all_cols.append(img_u8.reshape(-1, 3)[msk])

    if all_pts:
        pts_cat  = np.concatenate(all_pts,  axis=0)
        cols_cat = np.concatenate(all_cols, axis=0)
        rr.log("world/point_cloud", rr.Points3D(
            positions=pts_cat,
            colors=cols_cat,
            radii=0.003,
        ))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    rr.init("mast3r_dexycb", spawn=False)
    # FIX 3: correct connection method (matches view_gt_rerun.py)
    rr.connect_grpc(RERUN_ADDR)
    print(f"[rerun] streaming to {RERUN_ADDR} (gRPC)")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    views      = build_views(DATASET_ROOT)
    view_names = sorted(views.keys())
    n_frames   = len(views[view_names[0]])
    print(f"[INFO] {len(view_names)} views × {n_frames} frames  "
          f"→  {n_frames} independent reconstructions of {len(view_names)} images each")

    print(f"[INFO] loading model '{MODEL_NAME}' on {DEVICE} …")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    print("[INFO] model ready\n")

    # ── camera stabilization ──────────────────────────────────────────────────
    NUM_POSE_INIT_FRAMES  = 5
    cached_camera_params  = None

    cache_root = os.path.join(tempfile.gettempdir(), "mast3r_rerun_cache")
    os.makedirs(cache_root, exist_ok=True)

    for t in range(n_frames):
        print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")
        current_files = [views[v][t] for v in view_names]

        # Stage 1: estimate stable camera parameters once from the first N frames
        if cached_camera_params is None and t >= NUM_POSE_INIT_FRAMES - 1:
            print(f"[INFO] Stage 1: estimating camera parameters from first {NUM_POSE_INIT_FRAMES} frames …")
            calib_files = []
            for t_cal in range(NUM_POSE_INIT_FRAMES):
                calib_files.extend([views[v][t_cal] for v in view_names])

            calib_imgs  = load_images(calib_files, size=IMAGE_SIZE, verbose=True)
            calib_pairs = make_pairs(calib_imgs, scene_graph=SCENEGRAPH,
                                     prefilter=None, symmetrize=True)
            calib_cache = os.path.join(cache_root, "calibration")
            os.makedirs(calib_cache, exist_ok=True)

            calib_scene = sparse_global_alignment(
                calib_files, calib_pairs, calib_cache,
                model, lr1=LR1, niter1=NITER1, lr2=LR2, niter2=NITER2,
                device=DEVICE, opt_depth=OPT_DEPTH, shared_intrinsics=SHARED_INTRIN,
                matching_conf_thr=0.0,
            )

            all_intrinsics = to_numpy(calib_scene.intrinsics)
            all_poses      = to_numpy(calib_scene.cam2w)

            cached_camera_params = {}
            for i, vname in enumerate(view_names):
                indices = [i + (f * len(view_names)) for f in range(NUM_POSE_INIT_FRAMES)]
                K_avg   = np.mean(all_intrinsics[indices], axis=0)
                mid_idx = indices[NUM_POSE_INIT_FRAMES // 2]
                cached_camera_params[vname] = {
                    'intrinsics': torch.from_numpy(K_avg).to(DEVICE),
                    'cam2w':      torch.from_numpy(all_poses[mid_idx]).to(DEVICE),
                }
            print("[INFO] camera parameters cached.\n")

        # Stage 2: reconstruct with fixed cameras
        init_params = {}
        if cached_camera_params is not None:
            for i, v in enumerate(view_names):
                init_params[current_files[i]] = {
                    'intrinsics':       cached_camera_params[v]['intrinsics'],
                    'cam2w':            cached_camera_params[v]['cam2w'],
                    'freeze_pose':      True,
                    'freeze_intrinsics': True,
                }

        imgs  = load_images(current_files, size=IMAGE_SIZE, verbose=True)
        pairs = make_pairs(imgs, scene_graph=SCENEGRAPH, prefilter=None, symmetrize=True)

        cache_dir = os.path.join(cache_root, f"t{t:02d}")
        os.makedirs(cache_dir, exist_ok=True)

        scene = sparse_global_alignment(
            current_files, pairs, cache_dir,
            model,
            lr1=LR1,   niter1=NITER1,
            lr2=LR2,   niter2=NITER2,
            device=DEVICE,
            opt_depth=OPT_DEPTH,
            shared_intrinsics=SHARED_INTRIN,
            matching_conf_thr=0.0,
            init=init_params,
        )

        log_timestep(t, view_names, scene)
        print(f"  ✓ t={t:02d} logged to Rerun\n")

    print("[done] all timesteps streamed to Rerun.")


if __name__ == "__main__":
    main()