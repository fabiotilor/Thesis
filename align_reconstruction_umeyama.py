import argparse
import os
import tempfile
import numpy as np
import torch
import cv2
import rerun as rr
import glob
from collections import defaultdict

# MASt3R imports
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

# Umeyama alignment
from mast3r.utils.umeyama_alignment import estimate_similarity_transform, apply_similarity_transform

# ── configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"
DEPTH_SCALE = 0.001  # mm → metres
DEPTH_MAX_M = 2.0  # sentinel filter (matches view_gt_rerun.py)

LR1, NITER1 = 0.07, 300
LR2, NITER2 = 0.01, 300
MIN_CONF_THR = 1.5
SCENEGRAPH = "complete"
CLEAN_DEPTH = True
OPT_DEPTH = True
SHARED_INTRIN = False


# ── helpers ───────────────────────────────────────────────────────────────────

def get_masked_image(t: int, vname: str, rgb_path: str, mask_mode: str, cache_dir: str, dataset_root: str):
    if mask_mode == "none":
        return rgb_path

    view_dir = os.path.join(dataset_root, vname)
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


def build_views(dataset_root: str) -> dict:
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = defaultdict(list)
    for vd in sorted(glob.glob(os.path.join(dataset_root, "view_*"))):
        vname = os.path.basename(vd)
        rgb_dir = os.path.join(vd, "rgb")
        search = rgb_dir if os.path.isdir(rgb_dir) else vd
        frames = sorted(f for f in glob.glob(os.path.join(search, "*"))
                        if os.path.splitext(f.lower())[1] in img_exts)
        if frames:
            views[vname] = frames
    return dict(views)


def load_gt_params(view_dir: str):
    """
    Returns (K [3x3 float64], cam2world [4x4 float64]).
    Matches view_gt_rerun.py exactly:
      - K sliced to 3x3 (dataset stores 4x4)
      - extrinsics inverted: dataset stores world2cam, we need cam2world
    """
    data = np.load(os.path.join(view_dir, "intrinsics_extrinsics.npz"))
    K = data['intrinsics'].astype(np.float64)[:3, :3]  # FIX: slice to 3x3
    cam2world = np.linalg.inv(data['extrinsics'].astype(np.float64))  # FIX: invert world2cam
    return K, cam2world


def backproject(depth_m: np.ndarray, K: np.ndarray):
    """
    Unproject depth image (metres) to camera-space points.
    Matches view_gt_rerun.py exactly, including DEPTH_MAX_M sentinel filter.
    Returns (pts_cam [Nx3], mask [HxW bool]).
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = (depth_m > 0) & (depth_m < DEPTH_MAX_M)  # FIX: sentinel filter
    z = depth_m[mask]
    pts = np.stack([(u[mask] - cx) * z / fx,
                    (v[mask] - cy) * z / fy,
                    z], axis=-1)
    return pts, mask


def build_gt_pointcloud(t: int, view_names: list, dataset_root: str, subsample: int = 4, mask_mode: str = "none"):
    """
    Build a single fused GT point cloud for frame t across all cameras.
    """
    all_pts = []
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw * DEPTH_SCALE
        K, cam2world = load_gt_params(view_dir)
        pts_cam, orig_mask = backproject(depth_m, K)
        pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]

        if mask_mode != "none":
            mask_path = os.path.join(view_dir, "mask", f"{t:05d}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(view_dir, "mask", f"{t:06d}.png")
            if os.path.exists(mask_path):
                seg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if seg_mask.shape[:2] != depth_raw.shape[:2]:
                    seg_mask = cv2.resize(seg_mask, (depth_raw.shape[1], depth_raw.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                if mask_mode == "masked":
                    valid_seg = (seg_mask > 0)[orig_mask]
                else:
                    valid_seg = (seg_mask == 0)[orig_mask]
                pts_world = pts_world[valid_seg]

        all_pts.append(pts_world[::subsample])

    return np.concatenate(all_pts, axis=0) if all_pts else None


def build_gt_correspondences(t: int, view_names: list, dataset_root: str,
                             pts3d_list, confs, img_h: int, img_w: int, mask_mode: str = "none"):
    """
    Build pixel-aligned (GT, est) correspondence pairs for Umeyama.
    Both are sampled at the same valid pixels so the arrays are 1-to-1.
    """
    gt_corr = []
    est_corr = []

    for i, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m = cv2.resize(depth_raw, (img_w, img_h),
                             interpolation=cv2.INTER_NEAREST).astype(np.float32) * DEPTH_SCALE

        K, cam2world = load_gt_params(view_dir)

        # Adjust K for the resize
        K_r = K.copy()
        K_r[0, :] *= img_w / depth_raw.shape[1]
        K_r[1, :] *= img_h / depth_raw.shape[0]

        conf_i = confs[i]  # (H, W)
        est_world = pts3d_list[i].reshape(img_h, img_w, 3)  # (H, W, 3)

        valid = (depth_m > 0) & (depth_m < DEPTH_MAX_M) & (conf_i > MIN_CONF_THR)
        if valid.sum() < 100:
            continue

        vv, uu = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij')
        z = depth_m[valid]
        x = (uu[valid] - K_r[0, 2]) * z / K_r[0, 0]
        y = (vv[valid] - K_r[1, 2]) * z / K_r[1, 1]
        p_gt_cam = np.stack([x, y, z], axis=-1)
        p_gt_world = (cam2world[:3, :3] @ p_gt_cam.T).T + cam2world[:3, 3]

        gt_corr.append(p_gt_world)
        est_corr.append(est_world[valid])

    if not gt_corr:
        return None, None
    return np.concatenate(gt_corr, axis=0), np.concatenate(est_corr, axis=0)


def log_alignment_rerun(t: int, gt_pts, est_pts, aligned_pts):
    rr.set_time("timestep", sequence=t)
    if gt_pts is not None:
        rr.log("world/gt",
               rr.Points3D(positions=gt_pts, colors=[0, 255, 0], radii=0.002))
    if est_pts is not None:
        rr.log("world/estimated/raw",
               rr.Points3D(positions=est_pts, colors=[255, 0, 0], radii=0.002))
    if aligned_pts is not None:
        rr.log("world/estimated/aligned",
               rr.Points3D(positions=aligned_pts, colors=[0, 0, 255], radii=0.002))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_mode", type=str, choices=["none", "masked", "inverse_masked"], default="none")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    rr.init("mast3r_umeyama_alignment", spawn=False)
    rr.connect_grpc(RERUN_ADDR)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    views = build_views(DATASET_ROOT)
    view_names = sorted(views.keys())
    n_frames = len(views[view_names[0]])

    print(f"[INFO] loading model '{MODEL_NAME}' …")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)

    NUM_POSE_INIT_FRAMES = 5
    cached_camera_params = None
    cache_root = os.path.join(tempfile.gettempdir(), "mast3r_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    for t in range(n_frames):
        print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")
        current_files = [views[v][t] for v in view_names]

        # ── Stage 1: estimate stable cameras once from first N frames ──────
        if cached_camera_params is None and t >= NUM_POSE_INIT_FRAMES - 1:
            print(f"[INFO] Stage 1: estimating camera parameters …")
            calib_files = [
                get_masked_image(t_cal, v, views[v][t_cal], args.mask_mode, cache_root, DATASET_ROOT)
                for t_cal in range(NUM_POSE_INIT_FRAMES)
                for v in view_names
            ]
            calib_imgs = load_images(calib_files, size=IMAGE_SIZE)
            calib_pairs = make_pairs(calib_imgs, scene_graph=SCENEGRAPH, symmetrize=True)
            calib_scene = sparse_global_alignment(
                calib_files, calib_pairs, os.path.join(cache_root, "calib"),
                model, device=DEVICE, matching_conf_thr=0.0)

            all_K = to_numpy(calib_scene.intrinsics)
            all_pose = to_numpy(calib_scene.cam2w)
            cached_camera_params = {}
            for i, vname in enumerate(view_names):
                idx = [i + f * len(view_names) for f in range(NUM_POSE_INIT_FRAMES)]
                cached_camera_params[vname] = {
                    'intrinsics': torch.from_numpy(np.mean(all_K[idx], axis=0)).to(DEVICE),
                    'cam2w': torch.from_numpy(all_pose[idx[NUM_POSE_INIT_FRAMES // 2]]).to(DEVICE),
                }
            print("[INFO] camera parameters cached.\n")

        # ── Stage 2: fixed-camera reconstruction ───────────────────────────
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
                    'freeze_intrinsics': True,
                }

        imgs = load_images(masked_current_files, size=IMAGE_SIZE)
        pairs = make_pairs(imgs, scene_graph=SCENEGRAPH, symmetrize=True)
        scene = sparse_global_alignment(
            masked_current_files, pairs, os.path.join(cache_root, f"t{t:02d}"),
            model, device=DEVICE, matching_conf_thr=0.0, init=init_params)

        # ── collect estimated points (all cameras fused, subsampled) ───────
        pts3d_list, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH))
        img_h, img_w = confs[0].shape

        est_pts = np.concatenate([
            pts3d_list[i].reshape(-1, 3)[confs[i].ravel() > MIN_CONF_THR][::4]
            for i in range(len(view_names))
        ], axis=0)

        # ── collect GT points — ONE fused cloud per frame ──────────────────
        gt_pts = build_gt_pointcloud(t, view_names, DATASET_ROOT, subsample=4, mask_mode=args.mask_mode)

        if gt_pts is None:
            print(f"  ! t={t:02d}: no GT depth found, skipping")
            continue

        # ── pixel-aligned correspondences for Umeyama ──────────────────────
        gt_corr, est_corr = build_gt_correspondences(
            t, view_names, DATASET_ROOT, pts3d_list, confs, img_h, img_w, mask_mode=args.mask_mode)

        if gt_corr is not None:
            stride = max(1, len(gt_corr) // 5000)
            s, R, trans = estimate_similarity_transform(
                est_corr[::stride], gt_corr[::stride])
            aligned_pts = apply_similarity_transform(est_pts, s, R, trans)
            log_alignment_rerun(t, gt_pts, est_pts, aligned_pts)

            # Save aligned points for evaluation
            # Use appropriate out dir based on mask mode
            if args.mask_mode == "masked":
                out_dir = "aligned_outputs_masked"
            elif args.mask_mode == "inverse_masked":
                out_dir = "aligned_outputs_inverse"
            else:
                out_dir = "aligned_outputs"

            os.makedirs(out_dir, exist_ok=True)
            np.savez(f"{out_dir}/frame_{t:02d}.npz", gt_pts=gt_pts, aligned_pts=aligned_pts)

            print(f"  ✓ t={t:02d}  scale={s:.4f}  "
                  f"gt={len(gt_pts):,}  est={len(est_pts):,}  corr={len(gt_corr):,}")
        else:
            log_alignment_rerun(t, gt_pts, est_pts, None)
            print(f"  ! t={t:02d}: no valid correspondences for Umeyama")

    print("[done]")


if __name__ == "__main__":
    main()