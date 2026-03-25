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
from mast3r.utils.umeyama_alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
)
from mast3r.utils.optical_flow import stabilise_static_points, compute_static_mask

# Building Ground Truth
from mast3r.utils.gt import (
    build_gt_pointcloud,
    build_static_gt_pointcloud,
    get_static_correspondences,
    get_camera_correspondences,
    DEPTH_MAX_M,
)

# ── configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RERUN_ADDR = "rerun+http://127.0.0.1:9876/proxy"

MIN_CONF_THR = 2.0
SCENEGRAPH = "complete"
CLEAN_DEPTH = True


# ── helpers ───────────────────────────────────────────────────────────────────
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


def log_alignment_rerun(t, gt_pts, est_pts, aligned_pts, refined_pts=None, static_gt_pts=None):
    rr.set_time("timestep", sequence=t)
    if gt_pts is not None:
        rr.log("world/gt", rr.Points3D(positions=gt_pts, colors=[0, 255, 0], radii=0.002))
    if static_gt_pts is not None:
        rr.log("world/gt_static", rr.Points3D(positions=static_gt_pts, colors=[255, 165, 0], radii=0.002))
    if est_pts is not None:
        rr.log("world/estimated/raw", rr.Points3D(positions=est_pts, colors=[255, 0, 0], radii=0.002))
    if aligned_pts is not None:
        rr.log("world/estimated/aligned", rr.Points3D(positions=aligned_pts, colors=[0, 0, 255], radii=0.002))
    if refined_pts is not None:
        rr.log("world/estimated/stabilised",
               rr.Points3D(positions=refined_pts, colors=[255, 0, 255], radii=0.002))


def compute_all_static_masks(views, view_names, flow_threshold, verbose=True):
    print(f"\n[flow] Computing static masks (flow_threshold={flow_threshold}px)...")
    static_masks = {}
    for vname in view_names:
        mask = compute_static_mask(views[vname], flow_threshold)
        static_masks[vname] = mask
        if verbose:
            pct = 100 * mask.mean()
            print(f"  cam {vname}: {pct:.1f}% pixels static")
    return static_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_mode", choices=["none", "masked", "inverse_masked"], default="none")
    parser.add_argument("--stabilise", action="store_true",
                        help="Static regions → temporal median  → aligned_outputs_stabilised/")
    parser.add_argument("--flow_threshold", type=float, default=1.0,
                        help="Max flow magnitude (px) for a pixel to be classified static")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        rr.init("mast3r_stabilisation", spawn=False)
        rr.connect_grpc(RERUN_ADDR)
    except Exception as e:
        print(f"[WARN] Rerun init failed: {e}")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    target_views = ["05", "04", "00", "02"]
    views = build_views(DATASET_ROOT, target_views=target_views)
    view_names = sorted(views.keys())
    print(f"[INFO] Using views: {view_names}")

    n_frames = len(views[view_names[0]])
    print(f"[INFO] loading model '{MODEL_NAME}' …")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)

    cache_root = os.path.join(tempfile.gettempdir(), "mast3r_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)
    all_scenes = []

    for t in range(n_frames):
        print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")

        masked_current_files = [
            get_masked_image(t, v, views[v][t], args.mask_mode, cache_root, DATASET_ROOT)
            for v in view_names
        ]

        imgs = load_images(masked_current_files, size=IMAGE_SIZE)
        pairs = make_pairs(imgs, scene_graph=SCENEGRAPH, symmetrize=True)
        scene = sparse_global_alignment(
            masked_current_files,
            pairs,
            os.path.join(cache_root, f"t{t:02d}"),
            model,
            device=DEVICE,
            matching_conf_thr=0.0
        )

        if args.stabilise:
            all_scenes.append(scene)

        pts3d_list, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH))

        # ── Build full GT (used for evaluation and logging) ────────────────────
        gt_pts = build_gt_pointcloud(t, view_names, DATASET_ROOT,
                                     mask_mode=args.mask_mode)
        if gt_pts is None:
            continue

        # ── Build static-only GT (used for alignment only) ────────────────────
        static_gt_pts = build_static_gt_pointcloud(
            t, view_names, DATASET_ROOT,
            flow_threshold=args.flow_threshold,
            mask_mode=args.mask_mode,
        )

        # ── Point-based Umeyama: align via correspondences ────────────────────
        src_corr, dst_corr = get_static_correspondences(
            t, view_names, scene, DATASET_ROOT,
            flow_threshold=args.flow_threshold,
            min_conf_thr=MIN_CONF_THR
        )

        if src_corr is not None and len(src_corr) >= 3:
            s, R, tr = estimate_similarity_transform(src_corr, dst_corr)
            print(f"  ✓ t={t:02d}  scale={s:.4f}  corr={len(src_corr):,}")
        else:
            # Fallback: camera-based Umeyama if correspondences are too few
            print(f"  [WARN] t={t:02d}: too few correspondences ({len(src_corr) if src_corr is not None else 0}), falling back to camera-based")
            est_cam, gt_cam = get_camera_correspondences(t, view_names, scene, DATASET_ROOT)
            s, R, tr = estimate_similarity_transform(est_cam, gt_cam)
            print(f"  ✓ t={t:02d}  scale={s:.4f} (camera fallback)")

        # ── Build filtered estimated points (now aware of scale s) ─────────────────
        max_depth_est = DEPTH_MAX_M / s
        est_pts = np.concatenate([
            pts3d_list[i].reshape(-1, 3)[
                (confs[i].ravel() > MIN_CONF_THR) &
                (depthmaps[i].ravel() < max_depth_est)
            ]
            for i in range(len(view_names))
        ], axis=0)

        # ── Apply transform to full estimated points ───────────────────────────
        aligned_pts = apply_similarity_transform(est_pts, s, R, tr)

        # ── Log full GT + full estimated + full aligned + static GT ────────────
        log_alignment_rerun(t, gt_pts, est_pts, aligned_pts, static_gt_pts=static_gt_pts)

        out_dir = {"masked": "aligned_outputs_masked",
                   "inverse_masked": "aligned_outputs_inverse"}.get(
            args.mask_mode, "aligned_outputs")

        os.makedirs(out_dir, exist_ok=True)

        np.savez(os.path.join(out_dir, f"frame_{t:02d}.npz"),
                 gt_pts=gt_pts,
                 aligned_pts=aligned_pts,
                 scale=float(s),
                 frame_idx=int(t))

    # ── Post-processing passes ─────────────────────────────────────────────────
    if args.stabilise and all_scenes:
        base_dir = {"masked": "aligned_outputs_masked",
                    "inverse_masked": "aligned_outputs_inverse"}.get(args.mask_mode,
                                                                     "aligned_outputs")

        # Compute flow masks once — shared by all passes
        static_masks = compute_all_static_masks(views, view_names, args.flow_threshold)

        if args.stabilise:
            # Static regions only → aligned_outputs_stabilised/
            stabilise_static_points(
                views, view_names, all_scenes,
                get_cam_corr_fn=get_camera_correspondences,
                estimate_transform_fn=estimate_similarity_transform,
                log_rerun_fn=log_alignment_rerun,
                dataset_root=DATASET_ROOT,
                static_masks=static_masks,
                out_dir_in=base_dir,
                out_dir_out="aligned_outputs_stabilised",
                flow_threshold=args.flow_threshold)

    print("[done]")
    print("\nRun metrics with:")
    print(f"  python evaluate_temporal_consistency.py")
    if args.stabilise:
        print(f"  python evaluate_temporal_consistency.py --input_dir aligned_outputs_stabilised")

if __name__ == "__main__":
    main()