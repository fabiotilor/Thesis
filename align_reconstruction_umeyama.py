import os
import tempfile
import argparse
import json
import time
import numpy as np
import torch
try:
    import rerun as rr
except ImportError:
    rr = None
import glob
import cv2
from collections import defaultdict

# VGGT imports (replaces MASt3R/DUSt3R)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

# Umeyama alignment
from vggt.utils.umeyama_alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
)
from vggt.utils.optical_flow import compute_static_mask
from vggt.utils.camera_utils import get_rgb_path

# Building Ground Truth
from vggt.utils.gt import (
    load_gt_params,
    build_gt_pointcloud,
    build_static_gt_pointcloud,
    get_static_correspondences,
    get_camera_correspondences,
    build_gt_validity_masks,
    _load_hi4d_seg_mask,
    DEPTH_MAX_M,
)

# ── configuration ─────────────────────────────────────────────────────────────
from eval_config import (
    DATASETS, DATASET_BASE_ROOT, SUBJECT_NAMES, SUBJECT_BY_CODE,
    MODEL_NAME, IMAGE_SIZE, DEVICE, RERUN_ADDR,
    CONF_PERCENTILE, VIEW_CONFIGS, DEFAULT_TARGET_VIEWS, RERUN_EYE_UP
)
# NOTE: Import rerun logging lazily inside `run_reconstruction` to avoid
# circular-import issues when other modules import this file.
CLEAN_DEPTH = True
RUN_MULTI_VIEW_EVAL = True

# ── helpers ───────────────────────────────────────────────────────────────────
def get_masked_image(t, vname, rgb_path, cache_dir, dataset_root):
    return rgb_path


def build_views(dataset_root, target_views=None, dataset_type="dex-ycb"):
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = defaultdict(list)

    if dataset_type == "dex-ycb":
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
    elif dataset_type == "hi4d":
        # images/XX/000XXX.jpg
        img_dir = os.path.join(dataset_root, "images")
        if target_views:
            cam_ids = [str(v) for v in target_views]
        else:
            cam_ids = sorted(os.listdir(img_dir))

        for cid in cam_ids:
            cid_dir = os.path.join(img_dir, cid)
            if not os.path.isdir(cid_dir):
                continue
            frames = sorted(f for f in glob.glob(os.path.join(cid_dir, "*"))
                            if os.path.splitext(f.lower())[1] in img_exts)
            if frames:
                views[cid] = frames

    return dict(views)


def compute_all_static_masks(views, view_names, flow_threshold, verbose=True, dataset_type="dex-ycb"):
    print(f"\n[flow] Computing static masks (flow_threshold={flow_threshold}px)...")
    static_masks = {}
    for vname in view_names:
        mask = compute_static_mask(views[vname], dataset_type=dataset_type)
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
    skip_rerun_init=False,
    skip_existing_frames=True,
    use_gt_intrinsics=False,
    no_rerun=False,
    limit_frames=None,
    dataset_type="dex-ycb",
):
    rerun_stream = f"vggt_stabilisation_{run_tag}"
    if not skip_rerun_init and not no_rerun and rr is not None:
        try:
            rr.init(rerun_stream, spawn=False)
            rr.connect_grpc(RERUN_ADDR)
        except Exception as e:
            print(f"[WARN] Rerun init failed for {run_tag}: {e}")

    # Lazy import to avoid circular imports.
    from vggt.utils.rerun_logging import (
        configure_rerun_view_defaults,
        log_cameras_rerun,
        log_alignment_results,
    )
    # Log under the provided tag so run_full_pipeline can control the rerun hierarchy.
    log_root = f"{run_tag}"
    if not skip_rerun_init and not no_rerun and rr is not None:
        rr.log(log_root, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        eye_up = DATASETS.get(dataset_type, {}).get("eye_up", RERUN_EYE_UP)
        configure_rerun_view_defaults(log_root, eye_up)

    views = build_views(dataset_root, target_views=target_views, dataset_type=dataset_type)
    view_names = sorted(views.keys())
    if not view_names:
        print(f"[WARN] No valid views found for target_views={target_views}; skipping run.")
        return
    print(f"[INFO] Using views: {view_names}")

    n_frames = len(views[view_names[0]])
    if limit_frames:
        n_frames = min(n_frames, limit_frames)

    run_cache_root = os.path.join(cache_root, run_tag)
    os.makedirs(run_cache_root, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    run_start = time.perf_counter()
    frame_times_sec = []

    # Determine dtype for mixed-precision inference
    if DEVICE == "cuda" and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Determine frame indices
    t_indices = list(range(n_frames))
    if dataset_type == "hi4d":
        from eval_config import HI4D_START_FRAME, HI4D_STEP_SIZE, HI4D_TOTAL_FRAMES
        max_available = len(views[view_names[0]])
        target_count = limit_frames if limit_frames else HI4D_TOTAL_FRAMES
        t_indices = []
        curr = HI4D_START_FRAME
        while len(t_indices) < target_count and curr < max_available:
            t_indices.append(curr)
            curr += HI4D_STEP_SIZE

    for idx_i, t in enumerate(t_indices):
        # Extract actual frame index from filename
        first_view_path = views[view_names[0]][t]
        frame_filename = os.path.basename(first_view_path)
        actual_t = int(os.path.splitext(frame_filename)[0])

        if dataset_type == "hi4d":
            out_frame_path = os.path.join(out_dir, f"frame_{actual_t:06d}.npz")
        else:
            out_frame_path = os.path.join(out_dir, f"frame_{t:02d}.npz")

        if skip_existing_frames and os.path.exists(out_frame_path):
            print(f"  [SKIP] {run_tag}: existing {os.path.basename(out_frame_path)} found.")
            continue

        frame_start = time.perf_counter()
        try:
            print(f"── t={actual_t:06d} ({idx_i:02d} / {len(t_indices) - 1}) ──────────────────────────────────────")

            if not no_rerun and rr is not None:
                log_cameras_rerun(actual_t, view_names, dataset_root, log_root, dataset_type=dataset_type)

            # Collect current frame images across views
            current_files = [views[v][t] for v in view_names]

            # ── VGGT inference ──────────────────────────────────────────────────
            # For HI4D: apply segmentation masks to input images before inference
            if dataset_type == "hi4d":
                import tempfile as _tmpmod
                masked_files = []
                for i_v, (v, fpath) in enumerate(zip(view_names, current_files)):
                    seg_mask = _load_hi4d_seg_mask(dataset_root, v, actual_t)
                    if seg_mask is not None:
                        img = cv2.imread(fpath)
                        if img is not None:
                            # Resize mask to match image resolution
                            H_img, W_img = img.shape[:2]
                            if seg_mask.shape != (H_img, W_img):
                                seg_mask_resized = cv2.resize(
                                    seg_mask.astype(np.uint8), (W_img, H_img),
                                    interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
                            else:
                                seg_mask_resized = seg_mask
                            # Zero out background pixels
                            img[~seg_mask_resized] = 0
                            # Save to temp file
                            tmp_path = os.path.join(run_cache_root, f"masked_{v}_{actual_t:06d}.jpg")
                            cv2.imwrite(tmp_path, img)
                            masked_files.append(tmp_path)
                        else:
                            masked_files.append(fpath)
                    else:
                        masked_files.append(fpath)
                current_files = masked_files

            # load_and_preprocess_images returns (S, 3, H, W) tensor
            imgs = load_and_preprocess_images(current_files).to(DEVICE)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype) if DEVICE == "cuda" else torch.no_grad():
                    predictions = model(imgs)

            # Extract camera parameters from pose encoding
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], imgs.shape[-2:]
            )

            # Convert to numpy, squeeze batch dim
            pts3d_all = predictions["world_points"].cpu().numpy().squeeze(0)     # (V, H, W, 3)
            confs_all = predictions["world_points_conf"].cpu().numpy().squeeze(0) # (V, H, W)

            # Camera extrinsics/intrinsics: squeeze batch dim
            extrinsics_w2c = extrinsic.cpu().numpy().squeeze(0)  # (V, 3, 4) world-to-cam
            intrinsics_est = intrinsic.cpu().numpy().squeeze(0)  # (V, 3, 3)

            # Compute cam-to-world for est_poses
            est_c2w = closed_form_inverse_se3(extrinsic.squeeze(0)).cpu().numpy()  # (V, 4, 4)

            # ── Apply segmentation masks to output pointmaps for hi4d ──────────
            # This zeros out background points that the model may still output
            if dataset_type == "hi4d":
                for i_v, vname in enumerate(view_names):
                    seg_mask = _load_hi4d_seg_mask(dataset_root, vname, actual_t)
                    if seg_mask is not None:
                        H_out, W_out = confs_all[i_v].shape[:2]
                        mask_resized = cv2.resize(
                            seg_mask.astype(np.uint8), (W_out, H_out),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        pts3d_all[i_v][~mask_resized] = 0
                        confs_all[i_v][~mask_resized] = 0

            # ── [EXPERIMENT] GT Intrinsics Injection ────────────────────────────
            if use_gt_intrinsics and dataset_type == "dex-ycb":
                print(f"  [EXP] ── GT Intrinsics Back-Projection: t={actual_t} ──")
                V_inj, H_inj, W_inj = pts3d_all.shape[:3]
                SENSOR_H, SENSOR_W = 480, 640

                for i_v in range(V_inj):
                    K_est = intrinsics_est[i_v]
                    view_dir_i = os.path.join(dataset_root, view_names[i_v])
                    K_gt_native, _ = load_gt_params(view_dir_i, dataset_type=dataset_type)

                    sx = W_inj / SENSOR_W
                    sy = H_inj / SENSOR_H
                    K_gt_mod = K_gt_native.copy()
                    K_gt_mod[0, 0] *= sx; K_gt_mod[1, 1] *= sy
                    K_gt_mod[0, 2] *= sx; K_gt_mod[1, 2] *= sy

                    f_gt_x, f_gt_y = K_gt_mod[0, 0], K_gt_mod[1, 1]
                    cx_gt, cy_gt = K_gt_mod[0, 2], K_gt_mod[1, 2]

                    R_w2c = extrinsics_w2c[i_v, :3, :3]
                    t_w2c = extrinsics_w2c[i_v, :3, 3]
                    pts_world = pts3d_all[i_v].reshape(-1, 3)
                    pts_cam = (pts_world @ R_w2c.T) + t_w2c
                    Z = pts_cam[:, 2]

                    us = np.tile(np.arange(W_inj), H_inj)
                    vs = np.repeat(np.arange(H_inj), W_inj)
                    X_new = (us - cx_gt) / f_gt_x * Z
                    Y_new = (vs - cy_gt) / f_gt_y * Z
                    pts_cam_new = np.stack([X_new, Y_new, Z], axis=-1)

                    R_c2w = est_c2w[i_v, :3, :3]
                    t_c2w = est_c2w[i_v, :3, 3]
                    pts_world_new = (pts_cam_new @ R_c2w.T) + t_c2w

                    pts3d_all[i_v] = pts_world_new.reshape(H_inj, W_inj, 3)
                    intrinsics_est[i_v] = K_gt_mod

            # ── Full GT ─────────────────────────────────────────────────────────
            gt_pts = build_gt_pointcloud(
                actual_t, view_names, dataset_root, dataset_type=dataset_type
            )
            if gt_pts is None:
                print(f"  [WARN] No GT pointcloud at t={actual_t}; skipping frame.")
                continue

            # ── Correspondences ─────────────────────────────────────────────────
            src_corr, dst_corr = get_static_correspondences(
                actual_t, view_names, pts3d_all, confs_all, dataset_root,
                flow_threshold=flow_threshold,
                conf_percentile=CONF_PERCENTILE,
                use_static_mask=False,
                dataset_type=dataset_type
            )

            if src_corr is not None and len(src_corr) >= 3:
                s, R, tr = estimate_similarity_transform(src_corr, dst_corr)
                print(f"  ✓ t={actual_t}  scale={s:.4f}  corr={len(src_corr):,}")
            else:
                print(f"  [WARN] t={actual_t}: too few correspondences "
                      f"({len(src_corr) if src_corr is not None else 0}), falling back to camera-based")

                est_cam, gt_cam = get_camera_correspondences(
                    actual_t, view_names, est_c2w, dataset_root, dataset_type=dataset_type
                )
                s, R, tr = estimate_similarity_transform(est_cam, gt_cam)
                print(f"  ✓ t={actual_t}  scale={s:.4f} (camera fallback)")

            # ── Filter estimated points ─────────────────────────────────────────
            if dataset_type == "hi4d":
                # For Hi4D: seg masks already applied above; just filter by confidence
                est_pts_parts = []
                all_confs_flat = np.concatenate([c.ravel() for c in [confs_all[i] for i in range(confs_all.shape[0])]])
                frame_thr = np.quantile(all_confs_flat, 1.0 - CONF_PERCENTILE)
                for i_v in range(pts3d_all.shape[0]):
                    pts_i = pts3d_all[i_v].reshape(-1, 3)
                    conf_i = confs_all[i_v].ravel()
                    conf_ok = conf_i > frame_thr
                    est_pts_parts.append(pts_i[conf_ok])
            else:
                gt_validity_masks = build_gt_validity_masks(
                    actual_t, view_names, dataset_root,
                    depth_max_m=DEPTH_MAX_M,
                    target_hw=None,
                    dataset_type=dataset_type
                )

                est_pts_parts = []
                V = pts3d_all.shape[0]
                for i_v in range(V):
                    pts_i = pts3d_all[i_v].reshape(-1, 3)
                    conf_i = confs_all[i_v].ravel()
                    thr = np.percentile(conf_i, 100 * (1 - CONF_PERCENTILE))
                    conf_ok = conf_i > thr

                    gt_mask = gt_validity_masks[i_v]
                    if gt_mask is None:
                        vname = view_names[i_v] if i_v < len(view_names) else f"view_{i_v}"
                        print(f"  [WARN] no GT depth for {vname} at t={actual_t}, skipping view")
                        continue

                    H_mod, W_mod = confs_all[i_v].shape[:2]
                    if gt_mask.shape != (H_mod, W_mod):
                        gt_mask = cv2.resize(
                            gt_mask.astype(np.uint8), (W_mod, H_mod),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    valid = conf_ok & gt_mask.ravel()
                    est_pts_parts.append(pts_i[valid])

            est_pts = np.concatenate(est_pts_parts, axis=0) if est_pts_parts else np.empty((0, 3))
            aligned_pts = apply_similarity_transform(est_pts, s, R, tr)

            # ── Logging ─────────────────────────────────────────────────────────
            if not no_rerun and rr is not None:
                log_alignment_results(
                    actual_t, gt_pts, aligned_pts,
                    log_root=log_root,
                )
                time.sleep(0.01)

            # ── Collect Masks and Camera Params ─────────────────────────────────
            valid_masks = []
            valid_Ks = []
            valid_R_ts = []
            valid_est_poses = []
            valid_est_intrinsics = []

            for i_v, vname in enumerate(view_names):
                view_dir = os.path.join(dataset_root, vname)
                K, cam2world = load_gt_params(view_dir, dataset_type=dataset_type)
                R_t = np.linalg.inv(cam2world)

                if dataset_type == "hi4d":
                    # For Hi4D: use seg masks instead of flow masks
                    seg_mask = _load_hi4d_seg_mask(dataset_root, vname, actual_t)
                    if seg_mask is not None:
                        H_mod, W_mod = confs_all[i_v].shape[:2]
                        flow_mask = cv2.resize(
                            seg_mask.astype(np.uint8), (W_mod, H_mod),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    else:
                        H_mod, W_mod = confs_all[i_v].shape[:2]
                        flow_mask = np.ones((H_mod, W_mod), dtype=bool)
                else:
                    rgb_dir = os.path.join(view_dir, "rgb") if os.path.isdir(os.path.join(view_dir, "rgb")) else view_dir

                    def _rgb_path(frame_t):
                        for ext in (".png", ".jpg", ".jpeg"):
                            p = os.path.join(rgb_dir, f"{frame_t:05d}{ext}")
                            if os.path.exists(p):
                                return p
                        return None

                    rgb_t = _rgb_path(actual_t if dataset_type == "dex-ycb" else t)
                    rgb_adj = _rgb_path((actual_t if dataset_type == "dex-ycb" else t) + 1) or _rgb_path((actual_t if dataset_type == "dex-ycb" else t) - 1)
                    rgb_paths = [p for p in [rgb_t, rgb_adj] if p is not None]
                    flow_mask = compute_static_mask(rgb_paths, dataset_type=dataset_type)

                    if flow_mask is not None:
                        depth_path = os.path.join(view_dir, "depth", f"{actual_t if dataset_type == 'dex-ycb' else t:05d}.png")
                        if os.path.exists(depth_path):
                            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                            if flow_mask.shape != depth_raw.shape[:2]:
                                flow_mask = cv2.resize(
                                    flow_mask.astype(np.uint8),
                                    (depth_raw.shape[1], depth_raw.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(bool)
                    else:
                        print(f"  [WARN] {vname} t={actual_t}: flow mask failed, skipping view")
                        continue

                valid_masks.append(flow_mask)
                valid_Ks.append(K)
                valid_R_ts.append(R_t)
                valid_est_poses.append(est_c2w[i_v])
                valid_est_intrinsics.append(intrinsics_est[i_v])

            save_dict = {
                'gt_pts': gt_pts,
                'aligned_pts': aligned_pts,
                'scale': float(s),
                'R': R,
                'tr': tr,
                'pointmaps': pts3d_all,                  # (V, H, W, 3) — VGGT world_points
                'pointmaps_confs': confs_all,             # (V, H, W) — VGGT world_points_conf
                'frame_idx': int(actual_t),
                'Ks': np.array(valid_Ks),
                'R_ts': np.array(valid_R_ts),
                'est_poses': np.array(valid_est_poses),
                'est_intrinsics': np.array(valid_est_intrinsics),
            }
            if valid_masks:
                save_dict['masks_2d'] = np.stack(valid_masks)

            np.savez(out_frame_path, **save_dict)
            frame_times_sec.append(time.perf_counter() - frame_start)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] Frame t={t} failed: {e}")
            continue

    total_sec = time.perf_counter() - run_start
    timing_payload = {
        "strategy": os.path.basename(out_dir),
        "n_frames": int(len(t_indices)),
        "total_seconds": float(total_sec),
        "seconds_per_frame": float(total_sec / max(len(frame_times_sec), 1)),
        "frame_times_seconds": [float(v) for v in frame_times_sec],
    }
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, indent=2)
    print(
        f"[TIME] {timing_payload['strategy']}: total={timing_payload['total_seconds']:.2f}s  "
        f"per_frame={timing_payload['seconds_per_frame']:.3f}s"
    )


def parse_subject_selection_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all subjects (01..10).")
    for code in sorted(SUBJECT_BY_CODE.keys()):
        parser.add_argument(f"--{code}", dest=f"subject_{code}", action="store_true", help=f"Run subject {code}.")
    parser.add_argument(
        "--views",
        nargs="+",
        type=int,
        help="Optional view counts to run (e.g. --views 2 3 4). Defaults to [2,3,4] when multi-view eval is enabled.",
    )
    parser.add_argument(
        "--use-gt-intrinsics",
        action="store_true",
        help="Experiment: Use Ground Truth intrinsics to rescale VGGT pointmaps and poses.",
    )
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
    model = VGGT.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    cache_root = os.path.join(tempfile.gettempdir(), "vggt_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    selected_codes_str = ", ".join(name.split("subject-")[1][:2] for name in selected_subjects)
    print(f"[INFO] Selected subjects: {selected_codes_str}")

    for subject_name in selected_subjects:
        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_name)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found, skipping: {dataset_root}")
            continue

        print(f"\n[INFO] Processing subject: {subject_name}")
        camera_counts = args.views if args.views else ([2, 3, 4] if RUN_MULTI_VIEW_EVAL else [4])

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
                use_gt_intrinsics=args.use_gt_intrinsics,
            )

    print("[done]")
    print("\nRun metrics with:")
    print("  python evaluate_temporal_consistency.py")


if __name__ == "__main__":
    main()