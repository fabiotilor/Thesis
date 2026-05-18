import os
import tempfile
import argparse
import json
import time
import math
import numpy as np
import torch
import rerun as rr
import glob
import cv2
from collections import defaultdict

from PIL import Image
import torchvision.transforms as T
from pi3.utils.geometry import recover_intrinsic_from_rays_d

# Umeyama alignment
from pi3.utils.umeyama_alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
)
from pi3.utils.camera_utils import get_rgb_path
from pi3.utils.optical_flow import compute_static_mask

# Building Ground Truth
from pi3.utils.gt import (
    load_gt_params,
    build_gt_pointcloud,
    build_static_gt_pointcloud,
    get_static_correspondences,
    get_camera_correspondences,
    DEPTH_MAX_M,
    _load_hi4d_seg_mask,
)

# ── configuration ─────────────────────────────────────────────────────────────
from eval_config import (
    DATASETS, DATASET_BASE_ROOT, SUBJECT_NAMES, SUBJECT_BY_CODE,
    DEVICE, RERUN_ADDR,
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
        model_type,
        dataset_root,
        target_views,
        out_dir,
        cache_root,
        run_tag="default",
        skip_rerun_init=False,
        skip_existing_frames=True,
        no_rerun=False,
        limit_frames=None,
        dataset_type="dex-ycb",
):
    rerun_stream = f"pi3_stabilisation_{run_tag}"
    if not skip_rerun_init and not no_rerun:
        try:
            rr.init(rerun_stream, spawn=False)
            rr.connect_grpc(RERUN_ADDR)
        except Exception as e:
            print(f"[WARN] Rerun init failed for {run_tag}: {e}")

    # Lazy import to avoid circular imports.
    from pi3.utils.rerun_logging import (
        configure_rerun_view_defaults,
        log_cameras_rerun,
        log_alignment_results,
    )
    # Log under the provided tag so run_full_pipeline can control the rerun hierarchy.
    # When called from run_full_pipeline, rerun setup is already done there, so
    # avoid re-sending blueprints to reduce "overwriting" behavior.
    log_root = f"{run_tag}"
    if not skip_rerun_init and not no_rerun:
        rr.log(log_root, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        eye_up = DATASETS[dataset_type].get("eye_up", RERUN_EYE_UP)
        print(f"[RERUN] Using eye_up={eye_up} for dataset={dataset_type}")
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
    mask_base_dir = os.path.join(out_dir, "flow_masks")

    t_indices = range(n_frames)
    if dataset_type == "hi4d":
        # Start from frame 22 (index 21) with step size 1, process 24 frames
        start_idx = 21
        stride = 1
        max_available = len(views[view_names[0]])

        # Determine how many frames we actually want
        target_count = limit_frames if limit_frames else 24

        t_indices = []
        curr = start_idx
        while len(t_indices) < target_count and curr < max_available:
            t_indices.append(curr)
            curr += stride

    for i, t in enumerate(t_indices):
        # Extract actual frame index from filename if possible
        first_view_path = views[view_names[0]][t]
        frame_filename = os.path.basename(first_view_path)
        actual_t = int(os.path.splitext(frame_filename)[0])

        out_frame_path = os.path.join(out_dir, f"frame_{actual_t:06d}.npz")
        if skip_existing_frames and os.path.exists(out_frame_path):
            print(f"  [SKIP] {run_tag}: existing {os.path.basename(out_frame_path)} found.")
            continue

        frame_start = time.perf_counter()
        try:
            print(f"── t={actual_t:06d} ({i:02d} / {len(t_indices) - 1}) ──────────────────────────────────────")

            if not no_rerun:
                log_cameras_rerun(actual_t, view_names, dataset_root, log_root, dataset_type=dataset_type)

            masked_current_files = [
                get_masked_image(t, v, views[v][t], cache_root, dataset_root)
                for v in view_names
            ]

            imgs_list = []
            PIXEL_LIMIT = 255000

            # Determine target size once for the first image
            first_img = Image.open(masked_current_files[0]).convert('RGB')
            W_orig, H_orig = first_img.size
            scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
            W_target, H_target = W_orig * scale, H_orig * scale
            k, m = round(W_target / 14), round(H_target / 14)
            while (k * 14) * (m * 14) > PIXEL_LIMIT:
                if k / m > W_target / H_target:
                    k -= 1
                else:
                    m -= 1
            TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14

            for i, f in enumerate(masked_current_files):
                img = Image.open(f).convert('RGB')
                img_resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                img_np = np.array(img_resized)  # (H, W, 3)

                # Apply segmentation mask for hi4d to filter out background
                if dataset_type == "hi4d":
                    vname = view_names[i]
                    seg_mask = _load_hi4d_seg_mask(dataset_root, vname, actual_t)
                    if seg_mask is not None:
                        # Resize mask to match target resolution
                        mask_resized = cv2.resize(
                            seg_mask.astype(np.uint8),
                            (TARGET_W, TARGET_H),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        # Set background pixels to black
                        img_np[~mask_resized] = 0

                imgs_list.append(T.ToTensor()(Image.fromarray(img_np)))

            imgs_tensor = torch.stack(imgs_list).to(DEVICE)
            imgs_tensor = imgs_tensor.unsqueeze(0)  # (1, V, 3, H, W)

            if DEVICE == "cuda" and torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                autocast_kwargs = {"device_type": "cuda", "dtype": dtype}
            else:
                # MPS and CPU: run in float32 — half-precision SVD fallback crashes on CPU
                autocast_kwargs = None

            with torch.no_grad():
                if autocast_kwargs:
                    with torch.autocast(**autocast_kwargs):
                        if model_type == "pi3":
                            res = model(imgs_tensor)
                        else:
                            res = model(imgs=imgs_tensor, intrinsics=None, poses=None, depths=None)
                else:
                    if model_type == "pi3":
                        res = model(imgs_tensor)
                    else:
                        res = model(imgs=imgs_tensor, intrinsics=None, poses=None, depths=None)

            pts3d_np = res['points'][0].float().cpu().numpy()
            pts3d_list = [pts3d_np[i] for i in range(pts3d_np.shape[0])]

            confs_np = torch.sigmoid(res['conf'][0, ..., 0]).float().cpu().numpy()
            confs = [confs_np[i] for i in range(confs_np.shape[0])]

            # Apply segmentation masks to output pointmaps for hi4d
            # This zeros out background points that the model may still output
            if dataset_type == "hi4d":
                for i, vname in enumerate(view_names):
                    seg_mask = _load_hi4d_seg_mask(dataset_root, vname, actual_t)
                    if seg_mask is not None:
                        H, W = confs[i].shape[:2]
                        # Resize mask to match model output resolution
                        mask_resized = cv2.resize(
                            seg_mask.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        # Zero out background points in both pointmap and confidence
                        pts3d_list[i][~mask_resized] = 0
                        confs[i][~mask_resized] = 0

            rays_d = torch.nn.functional.normalize(res['local_points'], dim=-1)
            K_est = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
            est_intrinsics_all = K_est[0].float().cpu().numpy()

            # ── Compute Global Confidence Threshold for this Frame ─────────────
            all_confs = np.concatenate([c.ravel() for c in confs])
            frame_thr = np.quantile(all_confs, 1.0 - CONF_PERCENTILE)
            print(f"  [CONF] Global Frame Threshold (top {100 * CONF_PERCENTILE:.0f}%): {frame_thr:.4f}")

            est_poses_all = np.zeros((imgs_tensor.shape[1], 4, 4))
            for i in range(imgs_tensor.shape[1]):
                local_pts = res['local_points'][0, i].float().cpu().numpy().reshape(-1, 3)
                global_pts = res['points'][0, i].float().cpu().numpy().reshape(-1, 3)
                valid = confs[i].ravel() > frame_thr
                if np.sum(valid) > 10:
                    s_est, R_est, t_trans = estimate_similarity_transform(local_pts[valid], global_pts[valid])
                    T_mat = np.eye(4)
                    T_mat[:3, :3] = s_est * R_est
                    T_mat[:3, 3] = t_trans
                    est_poses_all[i] = T_mat
                else:
                    est_poses_all[i] = np.eye(4)

            # ── Precompute Flow Masks ──────────────────────────────────────────
            precomputed_masks = {}
            for i, vname in enumerate(view_names):
                view_dir_v = os.path.join(dataset_root, vname)
                rgb_t_v = get_rgb_path(view_dir_v, actual_t, dataset_type=dataset_type)
                rgb_adj_v = get_rgb_path(view_dir_v, actual_t + 1, dataset_type=dataset_type) or get_rgb_path(
                    view_dir_v, actual_t - 1, dataset_type=dataset_type)
                rgb_paths_v = [p for p in [rgb_t_v, rgb_adj_v] if p is not None]
                if len(rgb_paths_v) >= 2:
                    precomputed_masks[vname] = compute_static_mask(rgb_paths_v, dataset_type=dataset_type)
                else:
                    precomputed_masks[vname] = None

            # ── Full GT ─────────────────────────────────────────────────────────
            gt_pts = build_gt_pointcloud(
                actual_t, view_names, dataset_root, dataset_type=dataset_type
            )

            # ── Static GT ───────────────────────────────────────────────────────
            gt_static_pts = build_static_gt_pointcloud(
                actual_t, view_names, dataset_root,
                precomputed_masks=precomputed_masks,
                dataset_type=dataset_type
            )
            if gt_pts is None:
                print(f"  [WARN] No GT pointcloud at actual_t={actual_t}; skipping frame.")
                continue

            # ── Correspondences ─────────────────────────────────────────────────
            src_corr, dst_corr = get_static_correspondences(
                actual_t, view_names, pts3d_list, confs, dataset_root,
                conf_percentile=CONF_PERCENTILE,
                precomputed_masks=precomputed_masks,
                use_static_mask=False,
                dataset_type=dataset_type
            )

            if src_corr is not None and len(src_corr) >= 3:
                s, R, tr = estimate_similarity_transform(src_corr, dst_corr)
                print(f"  ✓ t={t:02d}  scale={s:.4f}  corr={len(src_corr):,}")
            else:
                print(f"  [WARN] t={t:02d}: too few correspondences "
                      f"({len(src_corr) if src_corr is not None else 0}), falling back to camera-based")

                est_cam, gt_cam = get_camera_correspondences(
                    actual_t, view_names, est_poses_all, dataset_root, dataset_type=dataset_type
                )
                s, R, tr = estimate_similarity_transform(est_cam, gt_cam)
                print(f"  ✓ t={t:02d}  scale={s:.4f} (camera fallback)")

            # ── Filter estimated points ─────────────────────────────────────────
            # Note: Background is already removed via input image masking,
            # so we only filter by confidence threshold here.
            est_pts_parts = []
            for i, vname in enumerate(view_names):
                pts_i = pts3d_list[i].reshape(-1, 3)
                conf_i = confs[i].ravel()
                conf_ok = conf_i > frame_thr
                est_pts_parts.append(pts_i[conf_ok])

            est_pts = np.concatenate(est_pts_parts, axis=0) if est_pts_parts else np.empty((0, 3))
            aligned_pts = apply_similarity_transform(est_pts, s, R, tr)

            # ── Logging ─────────────────────────────────────────────────────────
            if not no_rerun:
                log_alignment_results(
                    actual_t, gt_pts, aligned_pts,
                    gt_static_pts=gt_static_pts,
                    log_root=log_root,
                )

            # ── Collect camera params and flow masks ───────────────────────────
            valid_masks = []
            valid_Ks = []
            valid_R_ts = []
            valid_est_poses = []
            valid_est_intrinsics = []

            # FIX: intrinsics is a method call, not a plain attribute
            pass

            # ── Per-view flow masks ────────────────────────────────────────────
            for i, vname in enumerate(view_names):
                view_dir = os.path.join(dataset_root, vname)
                K, cam2world = load_gt_params(view_dir, dataset_type=dataset_type)
                R_t = np.linalg.inv(cam2world)

                flow_mask = precomputed_masks.get(vname)
                if flow_mask is None:
                    if dataset_type == "hi4d":
                        # Hi4D uses seg masks loaded in compute_static_mask; if unavailable, use all-True
                        H_mod, W_mod = confs[i].shape[:2]
                        flow_mask = np.ones((H_mod, W_mod), dtype=bool)
                    else:
                        print(
                            f"  [WARN] {vname} actual_t={actual_t}: not enough frames or flow mask failed, skipping view")
                        continue

                H_mod, W_mod = confs[i].shape[:2]
                flow_mask_mod = (
                    cv2.resize(flow_mask.astype(np.uint8), (W_mod, H_mod),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    if flow_mask.shape != (H_mod, W_mod) else flow_mask
                )

                # Save per-view mask
                view_mask_out = os.path.join(mask_base_dir, vname)
                os.makedirs(view_mask_out, exist_ok=True)
                cv2.imwrite(
                    os.path.join(view_mask_out, f"static_mask_{actual_t:06d}.png"),
                    flow_mask_mod.astype(np.uint8) * 255,
                )

                valid_masks.append(flow_mask_mod)
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
                'frame_idx': int(actual_t),
                'Ks': np.array(valid_Ks),
                'R_ts': np.array(valid_R_ts),
                'est_poses': np.array(valid_est_poses),
                'est_intrinsics': np.array(valid_est_intrinsics),
                'min_conf_thr': float(frame_thr),
                'conf_percentile': float(CONF_PERCENTILE)
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
    parser.add_argument("--model", type=str, choices=["pi3", "pi3x"], default="pi3", help="Model to evaluate")
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

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[INFO] loading model '{args.model}' …")
    if args.model == "pi3":
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained("yyfz233/Pi3").to(DEVICE).eval()
    elif args.model == "pi3x":
        from pi3.models.pi3x import Pi3X
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        model.disable_multimodal()
        model = model.to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    cache_root = os.path.join(tempfile.gettempdir(), "pi3_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    selected_codes_str = ", ".join(name.split("subject-")[1][:2] for name in selected_subjects)
    print(f"[INFO] Selected subjects: {selected_codes_str}")
    rr.init("pi3_stabilisation", spawn=False)
    rr.connect_grpc(RERUN_ADDR)
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
                model_type=args.model,
                dataset_root=dataset_root,
                target_views=target_views,
                out_dir=out_dir,
                cache_root=cache_root,
                run_tag=f"{args.model}_{subject_name}_{num_views}views",
            )

    print("[done]")
    print("\nRun metrics with:")
    print("  python evaluate_temporal_consistency.py")


if __name__ == "__main__":
    main()