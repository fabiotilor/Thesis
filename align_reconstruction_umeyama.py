import os
import tempfile
import argparse
import numpy as np
import torch
import rerun as rr
import glob
import cv2
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
from mast3r.utils.optical_flow import compute_static_mask

# Building Ground Truth
from mast3r.utils.gt import (
    load_gt_params,
    build_gt_pointcloud,
    build_static_gt_pointcloud,
    get_static_correspondences,
    get_camera_correspondences,
    build_gt_validity_masks,
    DEPTH_MAX_M,
)

# ── configuration ─────────────────────────────────────────────────────────────
from eval_config import (
    DATASET_BASE_ROOT, SUBJECT_NAMES, SUBJECT_BY_CODE,
    MODEL_NAME, IMAGE_SIZE, DEVICE, RERUN_ADDR,
    MIN_CONF_THR, VIEW_CONFIGS, DEFAULT_TARGET_VIEWS, SCENE_GRAPH, RERUN_EYE_UP
)
from mast3r.utils.rerun_logging import (
    configure_rerun_view_defaults,
    log_cameras_rerun,
    log_alignment_results
)
CLEAN_DEPTH = True
RUN_MULTI_VIEW_EVAL = True

# ── helpers ───────────────────────────────────────────────────────────────────
def get_masked_image(t, vname, rgb_path, cache_dir, dataset_root):
    return rgb_path


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


def run_reconstruction(
    model,
    dataset_root,
    target_views,
    out_dir,
    cache_root,
    flow_threshold=1.0,
    run_tag="default",
):
    rerun_stream = f"mast3r_stabilisation_{run_tag}"
    try:
        rr.init(rerun_stream, spawn=False)
        rr.connect_grpc(RERUN_ADDR)
    except Exception as e:
        print(f"[WARN] Rerun init failed for {run_tag}: {e}")
    log_root = f"world/{run_tag}"
    rr.log(log_root, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    configure_rerun_view_defaults(log_root, RERUN_EYE_UP)

    views = build_views(dataset_root, target_views=target_views)
    view_names = sorted(views.keys())
    if not view_names:
        print(f"[WARN] No valid views found for target_views={target_views}; skipping run.")
        return
    print(f"[INFO] Using views: {view_names}")

    n_frames = len(views[view_names[0]])
    run_cache_root = os.path.join(cache_root, run_tag)
    os.makedirs(run_cache_root, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for t in range(n_frames):
        print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")

        log_cameras_rerun(t, view_names, dataset_root, log_root)

        masked_current_files = [
            get_masked_image(t, v, views[v][t], cache_root, dataset_root)
            for v in view_names
        ]

        imgs = load_images(masked_current_files, size=IMAGE_SIZE)
        pairs = make_pairs(imgs, scene_graph=SCENE_GRAPH, symmetrize=True)
        scene = sparse_global_alignment(
            masked_current_files,
            pairs,
            os.path.join(run_cache_root, f"t{t:02d}"),
            model,
            device=DEVICE,
            matching_conf_thr=0.0
        )

        pts3d_list, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH))

        # ── Full GT ─────────────────────────────────────────────────────────
        gt_pts = build_gt_pointcloud(
            t, view_names, dataset_root
        )
        if gt_pts is None:
            continue

        # ── Correspondences ─────────────────────────────────────────────────
        src_corr, dst_corr = get_static_correspondences(
            t, view_names, scene, dataset_root,
            flow_threshold=flow_threshold,
            min_conf_thr=MIN_CONF_THR
        )

        if src_corr is not None and len(src_corr) >= 3:
            s, R, tr = estimate_similarity_transform(src_corr, dst_corr)
            print(f"  ✓ t={t:02d}  scale={s:.4f}  corr={len(src_corr):,}")
        else:
            print(f"  [WARN] t={t:02d}: too few correspondences "
                  f"({len(src_corr) if src_corr is not None else 0}), falling back to camera-based")

            est_cam, gt_cam = get_camera_correspondences(
                t, view_names, scene, dataset_root
            )
            s, R, tr = estimate_similarity_transform(est_cam, gt_cam)
            print(f"  ✓ t={t:02d}  scale={s:.4f} (camera fallback)")

        # ── Filter estimated points ─────────────────────────────────────────


        gt_validity_masks = build_gt_validity_masks(
            t, view_names, dataset_root,
            depth_max_m=DEPTH_MAX_M,
            target_hw=None,  # handled per-view below
        )

        est_pts_parts = []
        for i, vname in enumerate(view_names):
            pts_i = pts3d_list[i].reshape(-1, 3)
            conf_i = confs[i].ravel()
            conf_ok = conf_i > MIN_CONF_THR

            gt_mask = gt_validity_masks[i]
            if gt_mask is None:
                print(f"  [WARN] no GT depth for {vname} at t={t}, skipping view")
                continue

            # Use confs[i].shape — NOT pts3d_list[i].shape[:2] —
            # because pts3d_list may be at native resolution while
            # confs is at MASt3R output resolution (e.g. 384×512)
            H, W = confs[i].shape[:2]  # <-- fix is here
            if gt_mask.shape != (H, W):
                gt_mask = cv2.resize(
                    gt_mask.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            valid = conf_ok & gt_mask.ravel()  # now both (196608,)
            est_pts_parts.append(pts_i[valid])

        est_pts = np.concatenate(est_pts_parts, axis=0)

        # ── Apply alignment ─────────────────────────────────────────────────
        aligned_pts = apply_similarity_transform(est_pts, s, R, tr)

        # ── Logging ─────────────────────────────────────────────────────────
        log_alignment_results(
            t, gt_pts, aligned_pts,
            log_root=log_root,
        )

        # ── Collect Masks and Camera Params for Split-Accuracy Metrics (All Views) ──
        valid_masks = []
        valid_Ks = []
        valid_R_ts = []
        valid_est_poses = []
        valid_est_intrinsics = []

        try:
            im_poses = scene.get_im_poses()
        except AttributeError:
            im_poses = scene.get_poses()
        est_poses_all = to_numpy(im_poses)
        est_intrinsics_all = to_numpy(scene.intrinsics)

        for i, vname in enumerate(view_names):
            view_dir = os.path.join(dataset_root, vname)
            K, cam2world = load_gt_params(view_dir)
            R_t = np.linalg.inv(cam2world)

            rgb_dir = os.path.join(view_dir, "rgb") if os.path.isdir(os.path.join(view_dir, "rgb")) else view_dir

            def _rgb_path(frame_t):
                for ext in (".png", ".jpg", ".jpeg"):
                    p = os.path.join(rgb_dir, f"{frame_t:05d}{ext}")
                    if os.path.exists(p): return p
                return None

            rgb_t = _rgb_path(t)
            rgb_adj = _rgb_path(t + 1) or _rgb_path(t - 1)
            rgb_paths = [p for p in [rgb_t, rgb_adj] if p is not None]
            flow_mask = compute_static_mask(rgb_paths)

            if flow_mask is not None:
                depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
                if os.path.exists(depth_path):
                    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if flow_mask.shape != depth_raw.shape[:2]:
                        flow_mask = cv2.resize(flow_mask.astype(np.uint8), (depth_raw.shape[1], depth_raw.shape[0]),
                                               interpolation=cv2.INTER_NEAREST).astype(bool)
                valid_masks.append(flow_mask)
                valid_Ks.append(K)
                valid_R_ts.append(R_t)
                valid_est_poses.append(est_poses_all[i])
                valid_est_intrinsics.append(est_intrinsics_all[i])

        save_dict = {
            'gt_pts': gt_pts,
            'aligned_pts': aligned_pts,
            'scale': float(s),
            'R': R,
            'tr': tr,
            'pointmaps': np.stack(pts3d_list),
            'pointmaps_confs': np.stack(confs),
            'frame_idx': int(t),
            'Ks': np.array(valid_Ks),
            'R_ts': np.array(valid_R_ts),
            'est_poses': np.array(valid_est_poses),
            'est_intrinsics': np.array(valid_est_intrinsics)
        }
        if valid_masks:
            save_dict['masks_2d'] = np.stack(valid_masks)

        np.savez(os.path.join(out_dir, f"frame_{t:02d}.npz"), **save_dict)


def parse_subject_selection_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all subjects (01..10).")
    for code in sorted(SUBJECT_BY_CODE.keys()):
        parser.add_argument(f"--{code}", dest=f"subject_{code}", action="store_true", help=f"Run subject {code}.")
    return parser.parse_args()


def get_selected_subject_names(args):
    if args.all:
        return SUBJECT_NAMES

    selected_codes = [
        code for code in sorted(SUBJECT_BY_CODE.keys())
        if getattr(args, f"subject_{code}")
    ]
    if not selected_codes:
        selected_codes = ["01"]  # default selection
    return [SUBJECT_BY_CODE[code] for code in selected_codes]


def main():
    args = parse_subject_selection_args()
    selected_subjects = get_selected_subject_names(args)
    flow_threshold = 1.0

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[INFO] loading model '{MODEL_NAME}' …")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    cache_root = os.path.join(tempfile.gettempdir(), "mast3r_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    selected_codes_str = ", ".join(name.split("subject-")[1][:2] for name in selected_subjects)
    print(f"[INFO] Selected subjects: {selected_codes_str}")

    for subject_name in selected_subjects:
        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_name)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found, skipping: {dataset_root}")
            continue

        print(f"\n[INFO] Processing subject: {subject_name}")
        camera_counts = [2, 3, 4] if RUN_MULTI_VIEW_EVAL else [4]

        for num_views in camera_counts:
            target_views = VIEW_CONFIGS.get(num_views)
            if target_views is None:
                target_views = DEFAULT_TARGET_VIEWS
            out_dir = os.path.join("aligned_outputs", subject_name, f"{num_views}views")
            run_reconstruction(
                model=model,
                dataset_root=dataset_root,
                target_views=target_views,
                out_dir=out_dir,
                cache_root=cache_root,
                flow_threshold=flow_threshold,
                run_tag=f"{subject_name}_{num_views}views",
            )

    print("[done]")
    print("\nRun metrics with:")
    print("  python evaluate_temporal_consistency.py")


if __name__ == "__main__":
    main()