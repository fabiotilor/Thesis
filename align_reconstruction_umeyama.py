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
from mast3r.utils.optical_flow import stabilise_static_points

# ── configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"
DEPTH_SCALE = 0.001  # mm → metres
DEPTH_MAX_M = 2.0

LR1, NITER1 = 0.07, 300
LR2, NITER2 = 0.01, 300
MIN_CONF_THR = 1.5
SCENEGRAPH = "complete"
CLEAN_DEPTH = True
OPT_DEPTH = True
SHARED_INTRIN = False


# ── helpers ───────────────────────────────────────────────────────────────────
# To compare jitters between static and dynamic objects in the scene
def get_masked_image(t, vname, rgb_path, mask_mode, cache_dir, dataset_root):
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
        valid = (mask > 0) if mask_mode == "masked" else (mask == 0)
        img[~valid] = 0
    out_path = os.path.join(cache_dir, f"{vname}_{t:05d}_masked.jpg")
    cv2.imwrite(out_path, img)
    return out_path


def build_views(dataset_root, target_views=None):
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = defaultdict(list)
    dirs = ([os.path.join(dataset_root, f"view_{v}") for v in target_views]
            if target_views else sorted(glob.glob(os.path.join(dataset_root, "view_*"))))
    for vd in dirs:
        if not os.path.isdir(vd):
            continue
        vname = os.path.basename(vd)
        rgb_dir = os.path.join(vd, "rgb")
        search = rgb_dir if os.path.isdir(rgb_dir) else vd
        frames = sorted(f for f in glob.glob(os.path.join(search, "*"))
                        if os.path.splitext(f.lower())[1] in img_exts)
        if frames:
            views[vname] = frames
    return dict(views)


def load_gt_params(view_dir):
    data = np.load(os.path.join(view_dir, "intrinsics_extrinsics.npz"))
    K = data['intrinsics'].astype(np.float64)[:3, :3]
    cam2world = np.linalg.inv(data['extrinsics'].astype(np.float64))
    return K, cam2world


def backproject(depth_m, K):
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = (depth_m > 0) & (depth_m < DEPTH_MAX_M)
    z = depth_m[mask]
    pts = np.stack([(u[mask] - cx) * z / fx, (v[mask] - cy) * z / fy, z], axis=-1)
    return pts, mask


def build_gt_pointcloud(t, view_names, dataset_root, subsample=4, mask_mode="none"):
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
                seg = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if seg.shape[:2] != depth_raw.shape[:2]:
                    seg = cv2.resize(seg, (depth_raw.shape[1], depth_raw.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
                valid_seg = (seg > 0)[orig_mask] if mask_mode == "masked" else (seg == 0)[orig_mask]
                pts_world = pts_world[valid_seg]
        all_pts.append(pts_world[::subsample])
    return np.concatenate(all_pts, axis=0) if all_pts else None


def build_gt_correspondences(t, view_names, dataset_root, pts3d_list, confs,
                             img_h, img_w, mask_mode="none"):
    gt_corr, est_corr = [], []
    for i, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m = cv2.resize(depth_raw, (img_w, img_h),
                             interpolation=cv2.INTER_NEAREST).astype(np.float32) * DEPTH_SCALE
        K, cam2world = load_gt_params(view_dir)
        K_r = K.copy()
        K_r[0, :] *= img_w / depth_raw.shape[1]
        K_r[1, :] *= img_h / depth_raw.shape[0]
        conf_i = confs[i]
        est_world = pts3d_list[i].reshape(img_h, img_w, 3)
        valid = (depth_m > 0) & (depth_m < DEPTH_MAX_M) & (conf_i > MIN_CONF_THR)
        if valid.sum() < 100:
            continue
        vv, uu = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij')
        z = depth_m[valid]
        p_gt_world = (cam2world[:3, :3] @
                      np.stack([(uu[valid] - K_r[0, 2]) * z / K_r[0, 0],
                                (vv[valid] - K_r[1, 2]) * z / K_r[1, 1], z], axis=-1).T).T + cam2world[:3, 3]
        gt_corr.append(p_gt_world)
        est_corr.append(est_world[valid])
    if not gt_corr:
        return None, None
    return np.concatenate(gt_corr, axis=0), np.concatenate(est_corr, axis=0)


def get_camera_correspondences(t, view_names, scene, dataset_root):
    est_positions, gt_positions = [], []
    im_poses = scene.get_im_poses()
    for i, vname in enumerate(view_names):
        c2w_est = to_numpy(im_poses[i])
        est_positions.append(c2w_est[:3, 3])
        view_dir = os.path.join(dataset_root, vname)
        _, cam2world_gt = load_gt_params(view_dir)
        gt_positions.append(cam2world_gt[:3, 3])
    return np.array(est_positions), np.array(gt_positions)


def log_alignment_rerun(t, gt_pts, est_pts, aligned_pts, refined_pts=None):
    rr.set_time("timestep", sequence=t)
    if gt_pts is not None:
        rr.log("world/gt", rr.Points3D(positions=gt_pts, colors=[0, 255, 0], radii=0.002))
    if est_pts is not None:
        rr.log("world/estimated/raw", rr.Points3D(positions=est_pts, colors=[255, 0, 0], radii=0.002))
    if aligned_pts is not None:
        rr.log("world/estimated/aligned", rr.Points3D(positions=aligned_pts, colors=[0, 0, 255], radii=0.002))
    if refined_pts is not None:
        rr.log("world/estimated/stabilised",
               rr.Points3D(positions=refined_pts, colors=[255, 0, 255], radii=0.002))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_mode", choices=["none", "masked", "inverse_masked"], default="none")
    parser.add_argument("--stabilise", action="store_true", help="Run optical flow stabilisation")
    parser.add_argument("--flow_threshold", type=float, default=1.0, help="Max flow magnitude for static pixels")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    rr.init("mast3r_stabilisation", spawn=False)
    rr.connect_grpc(RERUN_ADDR)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    target_views = [f"{i:02d}" for i in range(8)]
    views = build_views(DATASET_ROOT, target_views=target_views)
    view_names = sorted(views.keys())
    print(f"[INFO] Using views: {view_names}")

    n_frames = len(views[view_names[0]])
    print(f"[INFO] loading model '{MODEL_NAME}' …")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)

    cached_camera_params = None
    cache_root = os.path.join(tempfile.gettempdir(), "mast3r_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    all_scenes = []

    for t in range(n_frames):
        print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")
        NUM_POSE_INIT_FRAMES = 5
        if cached_camera_params is None and t >= NUM_POSE_INIT_FRAMES - 1:
            calib_files = [get_masked_image(tc, v, views[v][tc], args.mask_mode, cache_root, DATASET_ROOT)
                           for tc in range(NUM_POSE_INIT_FRAMES) for v in view_names]
            calib_imgs = load_images(calib_files, size=IMAGE_SIZE)
            calib_pairs = make_pairs(calib_imgs, scene_graph=SCENEGRAPH, symmetrize=True)
            calib_scene = sparse_global_alignment(calib_files, calib_pairs, os.path.join(cache_root, "calib"),
                                                  model, device=DEVICE, matching_conf_thr=0.0)
            all_K, all_pose = to_numpy(calib_scene.intrinsics), to_numpy(calib_scene.cam2w)
            cached_camera_params = {}
            for i, v in enumerate(view_names):
                idx = [i + f * len(view_names) for f in range(NUM_POSE_INIT_FRAMES)]
                cached_camera_params[v] = {
                    'intrinsics': torch.from_numpy(np.mean(all_K[idx], axis=0)).to(DEVICE),
                    'cam2w': torch.from_numpy(all_pose[idx[NUM_POSE_INIT_FRAMES // 2]]).to(DEVICE),
                }

        masked_current_files = [get_masked_image(t, v, views[v][t], args.mask_mode, cache_root, DATASET_ROOT)
                                for v in view_names]
        init_params = {}
        if cached_camera_params:
            for v in view_names:
                init_params[get_masked_image(t, v, views[v][t], args.mask_mode, cache_root, DATASET_ROOT)] = {
                    'intrinsics': cached_camera_params[v]['intrinsics'],
                    'cam2w': cached_camera_params[v]['cam2w'],
                    'freeze_pose': True, 'freeze_intrinsics': True,
                }

        imgs = load_images(masked_current_files, size=IMAGE_SIZE)
        pairs = make_pairs(imgs, scene_graph=SCENEGRAPH, symmetrize=True)
        scene = sparse_global_alignment(masked_current_files, pairs, os.path.join(cache_root, f"t{t:02d}"),
                                        model, device=DEVICE, matching_conf_thr=0.0, init=init_params)

        if args.stabilise:
            all_scenes.append(scene)

        pts3d_list, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH))
        est_pts = np.concatenate([pts3d_list[i].reshape(-1, 3)[confs[i].ravel() > MIN_CONF_THR][::4]
                                  for i in range(len(view_names))], axis=0)
        gt_pts = build_gt_pointcloud(t, view_names, DATASET_ROOT, subsample=4, mask_mode=args.mask_mode)
        if gt_pts is None: continue

        est_cam, gt_cam = get_camera_correspondences(t, view_names, scene, DATASET_ROOT)
        s, R, tr = estimate_similarity_transform(est_cam, gt_cam)
        if t == 0:
            print(f"[DIAG] Camera Umeyama (t=0): scale={s:.4f}")

        aligned_pts = apply_similarity_transform(est_pts, s, R, tr)
        log_alignment_rerun(t, gt_pts, est_pts, aligned_pts)
        out_dir = {"masked": "aligned_outputs_masked", "inverse_masked": "aligned_outputs_inverse"}.get(args.mask_mode,
                                                                                                        "aligned_outputs")
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, f"frame_{t:02d}.npz"), gt_pts=gt_pts, aligned_pts=aligned_pts, scale=float(s),
                 frame_idx=int(t))
        print(f"  ✓ t={t:02d} scale={s:.4f} gt={len(gt_pts):,} est={len(est_pts):,}")

    if args.stabilise and all_scenes:
        out_dir_in = {"masked": "aligned_outputs_masked", "inverse_masked": "aligned_outputs_inverse"}.get(
            args.mask_mode, "aligned_outputs")
        out_dir_out = {"masked": "aligned_outputs_stabilised_masked",
                       "inverse_masked": "aligned_outputs_stabilised_inverse"}.get(args.mask_mode,
                                                                                   "aligned_outputs_stabilised")
        stabilise_static_points(views, view_names, all_scenes,
                                get_cam_corr_fn=get_camera_correspondences,
                                estimate_transform_fn=estimate_similarity_transform,
                                log_rerun_fn=log_alignment_rerun,
                                dataset_root=DATASET_ROOT,
                                out_dir_in=out_dir_in,
                                out_dir_out=out_dir_out,
                                flow_threshold=args.flow_threshold)

    print("[done]")


if __name__ == "__main__":
    main()