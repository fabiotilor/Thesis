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

# Building Ground Truth
from vggt.utils.gt import (
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
    CONF_PERCENTILE, VIEW_CONFIGS, DEFAULT_TARGET_VIEWS, RERUN_EYE_UP
)
# NOTE: Import rerun logging lazily inside `run_reconstruction` to avoid
# circular-import issues when other modules import this file.
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
    skip_rerun_init=False,
    skip_existing_frames=True,
    use_gt_intrinsics=False,
):
    rerun_stream = f"vggt_stabilisation_{run_tag}"
    if not skip_rerun_init:
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
    # When called from run_full_pipeline, rerun setup is already done there, so
    # avoid re-sending blueprints to reduce "overwriting" behavior.
    log_root = f"{run_tag}"
    if not skip_rerun_init:
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
    run_start = time.perf_counter()
    frame_times_sec = []

    # Determine dtype for mixed-precision inference
    if DEVICE == "cuda" and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    for t in range(n_frames):
        out_frame_path = os.path.join(out_dir, f"frame_{t:02d}.npz")
        if skip_existing_frames and os.path.exists(out_frame_path):
            print(f"  [SKIP] {run_tag}: existing {os.path.basename(out_frame_path)} found.")
            continue

        frame_start = time.perf_counter()
        try:
            print(f"── t={t:02d} / {n_frames - 1} ──────────────────────────────────────")

            log_cameras_rerun(t, view_names, dataset_root, log_root)

            # Collect current frame images across views
            current_files = [views[v][t] for v in view_names]

            # ── VGGT inference ──────────────────────────────────────────────────
            # load_and_preprocess_images returns (S, 3, H, W) tensor
            imgs = load_and_preprocess_images(current_files).to(DEVICE)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype) if DEVICE == "cuda" else torch.no_grad():
                    predictions = model(imgs)

            # Extract camera parameters from pose encoding
            # extrinsics: (B, S, 3, 4) world-to-camera; intrinsics: (B, S, 3, 3)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], imgs.shape[-2:]
            )

            # Convert to numpy, squeeze batch dim
            # world_points: (S, H, W, 3), world_points_conf: (S, H, W)
            pts3d_all = predictions["world_points"].cpu().numpy().squeeze(0)     # (V, H, W, 3)
            confs_all = predictions["world_points_conf"].cpu().numpy().squeeze(0) # (V, H, W)

            # Camera extrinsics/intrinsics: squeeze batch dim
            extrinsics_w2c = extrinsic.cpu().numpy().squeeze(0)  # (V, 3, 4) world-to-cam
            intrinsics_est = intrinsic.cpu().numpy().squeeze(0)  # (V, 3, 3)

            # Compute cam-to-world for est_poses (used for camera correspondences)
            # closed_form_inverse_se3 expects (N, 3, 4) or (N, 4, 4)
            est_c2w = closed_form_inverse_se3(extrinsic.squeeze(0)).cpu().numpy()  # (V, 4, 4)

            # ── [EXPERIMENT] GT Intrinsics Injection ────────────────────────────
            # Strategy: Instead of just scaling by f_gt/f_est (an approximation),
            # we do a proper back-projection:
            #   1. From the model's world_points and predicted w2c pose, recover
            #      the per-pixel depth in camera space: Z = (R_est @ P_world + t_est)[2]
            #   2. Back-project the depth through GT intrinsics:
            #      P_new = Z * K_gt_inv @ [u, v, 1]^T  (in camera space)
            #   3. Transform back to world space via est c2w:
            #      P_world_new = R_c2w @ P_new + t_c2w
            # This correctly handles both focal length and principal point differences.
            if use_gt_intrinsics:
                print(f"  [EXP] ── GT Intrinsics Back-Projection: t={t:02d} ──────────────────────")
                V_inj, H_inj, W_inj = pts3d_all.shape[:3]
                # Dex-YCB sensor resolution (the resolution at which K_gt is defined)
                SENSOR_H, SENSOR_W = 480, 640

                for i in range(V_inj):
                    K_est = intrinsics_est[i]         # (3,3) predicted intrinsics at 518px
                    f_est_x, f_est_y = K_est[0, 0], K_est[1, 1]
                    cx_est,  cy_est  = K_est[0, 2], K_est[1, 2]

                    # Load GT intrinsics at native sensor resolution
                    view_dir_i = os.path.join(dataset_root, view_names[i])
                    K_gt_native, _ = load_gt_params(view_dir_i)  # (3,3) at 640x480

                    # Scale GT K to model resolution (518x518)
                    sx = W_inj / SENSOR_W   # e.g. 518/640 = 0.809
                    sy = H_inj / SENSOR_H   # e.g. 518/480 = 1.079
                    K_gt_mod = K_gt_native.copy()
                    K_gt_mod[0, 0] *= sx   # fx
                    K_gt_mod[1, 1] *= sy   # fy
                    K_gt_mod[0, 2] *= sx   # cx
                    K_gt_mod[1, 2] *= sy   # cy

                    f_gt_x, f_gt_y = K_gt_mod[0, 0], K_gt_mod[1, 1]
                    cx_gt,  cy_gt  = K_gt_mod[0, 2], K_gt_mod[1, 2]

                    print(f"    View {view_names[i]} [{i}]:")
                    print(f"      EST K  -> fx={f_est_x:.2f}  fy={f_est_y:.2f}  cx={cx_est:.2f}  cy={cy_est:.2f}")
                    print(f"      GT  K  -> fx={f_gt_x:.2f}  fy={f_gt_y:.2f}  cx={cx_gt:.2f}  cy={cy_gt:.2f}")
                    print(f"      Scale ratio: sx_sensor={sx:.4f}  sy_sensor={sy:.4f}")
                    print(f"      Focal ratio: fx_gt/fx_est={f_gt_x/f_est_x:.4f}  fy_gt/fy_est={f_gt_y/f_est_y:.4f}")

                    # Step 1: Transform world_points into estimated camera space
                    # extrinsics_w2c[i] is (3, 4): [R | t]
                    R_w2c = extrinsics_w2c[i, :3, :3]  # (3,3)
                    t_w2c = extrinsics_w2c[i, :3,  3]  # (3,)

                    pts_world = pts3d_all[i].reshape(-1, 3)  # (H*W, 3)
                    pts_cam   = (pts_world @ R_w2c.T) + t_w2c  # (H*W, 3) in camera space

                    # Step 2: Extract per-pixel depth (Z > 0 is in front of camera)
                    Z = pts_cam[:, 2]  # (H*W,)

                    # Diagnostic: depth stats before injection
                    valid_depth = Z > 0
                    print(f"      Depth (est cam space): min={Z[valid_depth].min():.4f}  "
                          f"max={Z[valid_depth].max():.4f}  "
                          f"mean={Z[valid_depth].mean():.4f}  "
                          f"valid_px={valid_depth.sum():,}/{len(Z):,}")

                    # Step 3: Build pixel grid [u, v] at model resolution
                    us = np.tile(np.arange(W_inj), H_inj)    # (H*W,)
                    vs = np.repeat(np.arange(H_inj), W_inj)  # (H*W,)

                    # Step 4: Back-project through GT intrinsics
                    # X_cam_new = (u - cx_gt) / fx_gt * Z
                    # Y_cam_new = (v - cy_gt) / fy_gt * Z
                    # Z_cam_new = Z  (depth is preserved)
                    X_new = (us - cx_gt) / f_gt_x * Z
                    Y_new = (vs - cy_gt) / f_gt_y * Z
                    pts_cam_new = np.stack([X_new, Y_new, Z], axis=-1)  # (H*W, 3)

                    # Step 5: Transform corrected camera-space points back to world space
                    # c2w = inv(w2c), est_c2w[i] is (4,4)
                    R_c2w = est_c2w[i, :3, :3]  # (3,3)
                    t_c2w = est_c2w[i, :3,  3]  # (3,)
                    pts_world_new = (pts_cam_new @ R_c2w.T) + t_c2w  # (H*W, 3)

                    # Diagnostic: compare old vs new point cloud centroid
                    old_centroid = pts_world[valid_depth].mean(axis=0)
                    new_centroid = pts_world_new[valid_depth].mean(axis=0)
                    print(f"      Centroid (old world): {old_centroid}")
                    print(f"      Centroid (new world): {new_centroid}")
                    centroid_shift = np.linalg.norm(new_centroid - old_centroid)
                    print(f"      Centroid shift (m):   {centroid_shift:.5f}")

                    # Write corrected points back
                    pts3d_all[i] = pts_world_new.reshape(H_inj, W_inj, 3)

                    # Step 6: Update intrinsics record to GT (for saving/evaluation)
                    intrinsics_est[i] = K_gt_mod
                    print(f"      intrinsics_est[{i}] updated to GT K (at model res).")

                print(f"  [EXP] ── Injection complete for t={t:02d} ────────────────────────────")

            # ── Full GT ─────────────────────────────────────────────────────────
            gt_pts = build_gt_pointcloud(
                t, view_names, dataset_root
            )
            if gt_pts is None:
                print(f"  [WARN] No GT pointcloud at t={t}; skipping frame.")
                continue

            # ── Correspondences ─────────────────────────────────────────────────
            # Changed for VGGT: pass raw pointmaps and confs instead of scene
            src_corr, dst_corr = get_static_correspondences(
                t, view_names, pts3d_all, confs_all, dataset_root,
                flow_threshold=flow_threshold,
                conf_percentile=CONF_PERCENTILE,
                use_static_mask=False
            )

            if src_corr is not None and len(src_corr) >= 3:
                s, R, tr = estimate_similarity_transform(src_corr, dst_corr)
                print(f"  ✓ t={t:02d}  scale={s:.4f}  corr={len(src_corr):,}")
            else:
                print(f"  [WARN] t={t:02d}: too few correspondences "
                      f"({len(src_corr) if src_corr is not None else 0}), falling back to camera-based")

                # Changed for VGGT: pass est_poses array instead of scene
                est_cam, gt_cam = get_camera_correspondences(
                    t, view_names, est_c2w, dataset_root
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
            V = pts3d_all.shape[0]
            for i in range(V):
                # VGGT pointmaps are already (H, W, 3)
                pts_i = pts3d_all[i].reshape(-1, 3)
                conf_i = confs_all[i].ravel()
                # Filter top-K percentile
                thr = np.percentile(conf_i, 100 * (1 - CONF_PERCENTILE))
                conf_ok = conf_i > thr

                gt_mask = gt_validity_masks[i]
                if gt_mask is None:
                    vname = view_names[i] if i < len(view_names) else f"view_{i}"
                    print(f"  [WARN] no GT depth for {vname} at t={t}, skipping view")
                    continue

                # Match GT mask resolution to model output
                H_mod, W_mod = confs_all[i].shape[:2]
                if gt_mask.shape != (H_mod, W_mod):
                    gt_mask = cv2.resize(
                        gt_mask.astype(np.uint8), (W_mod, H_mod),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                valid = conf_ok & gt_mask.ravel()
                est_pts_parts.append(pts_i[valid])

            est_pts = np.concatenate(est_pts_parts, axis=0)
            aligned_pts = apply_similarity_transform(est_pts, s, R, tr)

            # ── Logging ─────────────────────────────────────────────────────────
            log_alignment_results(
                t, gt_pts, aligned_pts,
                log_root=log_root,
            )
            time.sleep(0.01) #ensure logs are flushed, VGGT is too fast.
            # ── Collect Masks and Camera Params for Split-Accuracy Metrics (All Views) ──
            valid_masks = []
            valid_Ks = []
            valid_R_ts = []
            valid_est_poses = []
            valid_est_intrinsics = []

            for i, vname in enumerate(view_names):
                view_dir = os.path.join(dataset_root, vname)
                K, cam2world = load_gt_params(view_dir)
                R_t = np.linalg.inv(cam2world)

                rgb_dir = os.path.join(view_dir, "rgb") if os.path.isdir(os.path.join(view_dir, "rgb")) else view_dir

                def _rgb_path(frame_t):
                    for ext in (".png", ".jpg", ".jpeg"):
                        p = os.path.join(rgb_dir, f"{frame_t:05d}{ext}")
                        if os.path.exists(p):
                            return p
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
                            flow_mask = cv2.resize(
                                flow_mask.astype(np.uint8),
                                (depth_raw.shape[1], depth_raw.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                    valid_masks.append(flow_mask)
                    valid_Ks.append(K)
                    valid_R_ts.append(R_t)
                    valid_est_poses.append(est_c2w[i])
                    valid_est_intrinsics.append(intrinsics_est[i])

            save_dict = {
                'gt_pts': gt_pts,
                'aligned_pts': aligned_pts,
                'scale': float(s),
                'R': R,
                'tr': tr,
                'pointmaps': pts3d_all,                  # (V, H, W, 3) — VGGT world_points
                'pointmaps_confs': confs_all,             # (V, H, W) — VGGT world_points_conf
                'frame_idx': int(t),
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
            print(f"  [ERROR] Frame t={t} failed: {e}")
            continue

    total_sec = time.perf_counter() - run_start
    timing_payload = {
        "strategy": os.path.basename(out_dir),
        "n_frames": int(n_frames),
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