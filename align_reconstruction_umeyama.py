#!/usr/bin/env python3
"""
VGGT4D 4D Reconstruction (All-At-Once).

Process all frames and views natively through the model.
Finds a single global Umeyama registration across time.
"""
import gc
import os
import tempfile
import argparse
import json
import math
import time
import glob
import traceback
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rerun as rr
from einops import rearrange

# ── VGGT4D model & inference ──────────────────────────────────────────────────
# ── VGGT4D model & inference ──────────────────────────────────────────────────
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.utils.model_utils import run_vggt4d_3stage_inference

# ── Shared utilities ──────────────────────────────────────────────────────────
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.umeyama_alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
)
from vggt.utils.gt import (
    load_gt_params,
    build_gt_pointcloud,
    build_static_gt_pointcloud,
    get_camera_correspondences,
    get_single_view_correspondences,
    build_gt_validity_masks,
    DEPTH_MAX_M,
)
from vggt.utils.camera_utils import build_views
from vggt.utils.alignment_4d import (
    generate_windows,
    gaussian_temporal_weights,
    FusionAccumulator,
)
from vggt.utils.rerun_logging import (
    configure_rerun_view_defaults,
    log_cameras_rerun,
    log_estimated_cameras_rerun,
    log_pointcloud,
    log_raw_window,
    log_fused_frame,
)

# ── Configuration ─────────────────────────────────────────────────────────────
from eval_config import (
    DATASET_BASE_ROOT,
    SUBJECT_NAMES,
    SUBJECT_BY_CODE,
    VGGT4D_CHECKPOINT,
    IMAGE_SIZE,
    DEVICE,
    RERUN_ADDR,
    CONF_PERCENTILE,
    VIEW_CONFIGS,
    DEFAULT_TARGET_VIEWS,
    RERUN_EYE_UP,
    get_dataset_config,
    get_subject_by_code,
    get_view_config,
)

CLEAN_DEPTH = True
RUN_MULTI_VIEW_EVAL = True


def run_all_at_once_pipeline(
        model, views, dataset_root, out_dir, run_tag, log_root,
        skip_existing_frames=True,
        no_rerun=False,
        dataset_type="dex-ycb",
):
    """
    Process ALL views × ALL frames in a single VGGT4D forward pass.

    Since VGGT4D sees every view simultaneously, its outputs are already
    in a shared world coordinate frame. There is no per-view alignment —
    we merge all views' pointmaps per frame into one pointcloud, solve
    ONE global Umeyama, and apply it uniformly.

    Rerun streams:
        <log_root>/raw/pointcloud      — merged raw model output (orange)
        <log_root>/aligned/pointcloud  — merged after Umeyama (blue)
        <log_root>/gt/pointcloud       — ground truth (green)
    """
    from vggt.utils.rerun_logging import (
        log_cameras_rerun,
        log_estimated_cameras_rerun,
        log_pointcloud,
    )

    view_names = sorted(views.keys())
    V = len(view_names)
    n_frames = len(views[view_names[0]])

    # ── 1. Build interleaved sequence (frame-major) ──────────────────────
    print(f"\n[ALL-AT-ONCE]  V={V}  T={n_frames}  total_images={V * n_frames}")
    seq_paths = []
    for fi in range(n_frames):
        for vname in view_names:
            seq_paths.append(views[vname][fi])

    # ── 2. HI4D Input Masking ───────────────────────────────────────────
    # Masking input images helps the model focus on humans and improves scale accuracy.
    if dataset_type == "hi4d":
        from vggt.utils.gt import _load_hi4d_seg_mask
        print("[INFO] Pre-masking HI4D input images...")
        masked_seq_paths = []
        cache_dir = os.path.join(tempfile.gettempdir(), "vggt4d_hi4d_masked")
        os.makedirs(cache_dir, exist_ok=True)

        for i, fpath in enumerate(seq_paths):
            # Frame t is i // V, View v is i % V
            t = i // V
            vname = view_names[i % V]
            actual_t = int(os.path.splitext(os.path.basename(fpath))[0])

            seg_mask = _load_hi4d_seg_mask(dataset_root, vname, actual_t)
            if seg_mask is not None:
                img = cv2.imread(fpath)
                if img is not None:
                    H_img, W_img = img.shape[:2]
                    mask_resized = cv2.resize(
                        seg_mask.astype(np.uint8), (W_img, H_img),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    img[~mask_resized] = 0

                    tmp_path = os.path.join(cache_dir, f"masked_{actual_t:06d}_{vname}.jpg")
                    cv2.imwrite(tmp_path, img)
                    masked_seq_paths.append(tmp_path)
                else:
                    masked_seq_paths.append(fpath)
            else:
                masked_seq_paths.append(fpath)
        seq_paths = masked_seq_paths

    # ── 3. Run VGGT4D 3-stage inference ──────────────────────────────────
    chunk = run_vggt4d_3stage_inference(model, seq_paths, DEVICE)
    # chunk keys: world_points (T*V, H, W, 3), world_points_conf (T*V, H, W),
    #             dynamic_masks (T*V, H, W), cam2world (T*V,4,4), etc.

    H, W = chunk["world_points"].shape[1:3]

    # ── 3. De-interleave: (T*V, ...) → per-frame, per-view ──────────────
    #   Index layout: [v0_f0, v1_f0, ..., vV_f0, v0_f1, v1_f1, ..., vV_fT]
    #   For frame t, images are at indices [t*V, t*V+1, ..., t*V+V-1]
    world_points = chunk["world_points"]  # (T*V, H, W, 3)
    world_confs = chunk["world_points_conf"]  # (T*V, H, W)
    dyn_masks = chunk["dynamic_masks"]  # (T*V, H, W) bool
    cam2world = chunk["cam2world"]  # (T*V, 4, 4)
    extrinsic = chunk["extrinsic"]  # (T*V, 3, 4)
    intrinsic = chunk["intrinsic"]  # (T*V, 3, 3)

    # ── 4. Collect Per-Frame Correspondences & Solve Umeyama ──────────────
    print("[INFO] Collecting per-frame correspondences and solving Umeyama...")
    per_frame_transforms = []  # list of (s, R, tr) per frame

    # Per-frame raw merged pointclouds and metadata (stored for Rerun/NPZ)
    per_frame_raw_pts = []  # list of (N_t, 3)
    per_frame_raw_confs = []  # list of (N_t,)
    per_frame_pointmaps = []  # (V, H, W, 3)
    per_frame_confs = []  # (V, H, W)
    per_frame_masks = []  # (V, H, W) bool
    per_frame_cams = []  # dict per frame

    for t in range(n_frames):
        idx_start = t * V
        idx_end = idx_start + V

        # Get actual frame ID from filename (important for HI4D indexing)
        first_view_path = views[view_names[0]][t]
        actual_t = int(os.path.splitext(os.path.basename(first_view_path))[0])

        # Per-view pointmaps for this frame (V images in unified coords)
        pm_views = world_points[idx_start:idx_end]  # (V, H, W, 3)
        conf_views = world_confs[idx_start:idx_end]  # (V, H, W)
        dyn_views = dyn_masks[idx_start:idx_end]  # (V, H, W)

        per_frame_pointmaps.append(pm_views)
        per_frame_confs.append(conf_views)
        per_frame_masks.append(~dyn_views)  # True = static

        per_frame_cams.append({
            "cam2world": cam2world[idx_start:idx_end],
            "extrinsic": extrinsic[idx_start:idx_end],
            "intrinsic": intrinsic[idx_start:idx_end],
            "actual_t": actual_t,
        })

        # ── Compute Global Confidence Threshold for this Frame ─────────────
        all_confs_f = np.concatenate([c.ravel() for c in conf_views])
        frame_thr = np.quantile(all_confs_f, 1.0 - CONF_PERCENTILE)

        # ── Apply segmentation masks to output pointmaps for hi4d ──────────
        # This zeros out background points that the model may still output
        # Uses crop-aware resize to match the model's center-crop preprocessing
        if dataset_type == "hi4d":
            from vggt.utils.gt import _load_hi4d_seg_mask, _resize_mask_crop_aware
            for i_v, vname in enumerate(view_names):
                seg_mask = _load_hi4d_seg_mask(dataset_root, vname, actual_t)
                if seg_mask is not None:
                    H_out, W_out = conf_views[i_v].shape[:2]
                    H_orig, W_orig = seg_mask.shape[:2]
                    mask_resized = _resize_mask_crop_aware(
                        seg_mask, W_orig, H_orig, H_out, W_out
                    )
                    pm_views[i_v][~mask_resized] = 0
                    conf_views[i_v][~mask_resized] = 0

        # ── Collect Point Correspondences & Align Per-View ─────────────────
        est_pts_parts = []
        raw_conf_parts = []

        for vi, vname in enumerate(view_names):
            pts_flat = pm_views[vi].reshape(-1, 3)
            conf_flat = conf_views[vi].ravel()
            static = ~dyn_views[vi]
            H_out, W_out = conf_views[vi].shape[:2]

            # Collect correspondences (model pts → GT world pts)
            # This uses pixel-wise projection for HI4D, which is scale-invariant.
            src, dst = get_single_view_correspondences(
                actual_t, vname, pm_views[vi], conf_views[vi],
                dataset_root,
                static_mask=static,
                conf_percentile=CONF_PERCENTILE,
                use_static_mask=False,
                dataset_type=dataset_type,
            )

            # Filter valid points
            # For HI4D, add minimum confidence threshold to filter noise even at 100% confidence
            min_conf_thresh = 0.01 if dataset_type == "hi4d" else 0.0
            valid = (conf_flat > frame_thr) & (conf_flat > min_conf_thresh)
            gt_validity = build_gt_validity_masks(
                actual_t, [vname], dataset_root,
                depth_max_m=DEPTH_MAX_M, target_hw=(H_out, W_out),
                dataset_type=dataset_type,
            )
            if gt_validity[0] is not None:
                gt_mask = gt_validity[0]
                if gt_mask.shape != (H_out, W_out):
                    gt_mask = cv2.resize(
                        gt_mask.astype(np.uint8), (W_out, H_out),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                valid &= gt_mask.ravel()

            if not valid.any():
                continue

            raw_pts = pts_flat[valid]
            raw_conf = conf_flat[valid]

            # Solve Umeyama per-view and apply it independently (matches MASt3R)
            if src is not None and len(src) >= 3:
                s_v, R_v, tr_v = estimate_similarity_transform(src, dst)
                aligned_v = apply_similarity_transform(raw_pts, s_v, R_v, tr_v)
                est_pts_parts.append(aligned_v)
                raw_conf_parts.append(raw_conf)
            else:
                print(f"  [WARN] t={actual_t} view={vname}: Not enough correspondences; skipping view.")

        # ── 5. Global Transform (Bypassed since we align per-view) ────────
        # Since we applied the transform per-view, the global transform is Identity
        s, R, tr = 1.0, np.eye(3), np.zeros(3)
        per_frame_transforms.append((s, R, tr))

        aligned_merged = (
            np.concatenate(est_pts_parts) if est_pts_parts
            else np.zeros((0, 3), dtype=np.float32)
        )
        aligned_merged_conf = (
            np.concatenate(raw_conf_parts) if raw_conf_parts
            else np.zeros((0,), dtype=np.float32)
        )

        # We store the aligned points in per_frame_raw_pts because the downstream NPZ
        # saving will apply the (Identity) transform to them
        per_frame_raw_pts.append(aligned_merged)
        per_frame_raw_confs.append(aligned_merged_conf)

    # ── 6. Save per-frame NPZ with aligned results ─────────────────────
    # Free GPU memory
    del chunk, world_points, world_confs, dyn_masks
    torch.cuda.empty_cache()
    gc.collect()

    frame_times_sec = []
    for t in range(n_frames):
        out_frame_path = os.path.join(out_dir, f"frame_{t:02d}.npz")
        if skip_existing_frames and os.path.exists(out_frame_path):
            print(f"  [SKIP] {run_tag}: existing {os.path.basename(out_frame_path)}")
            continue

        frame_start = time.perf_counter()
        try:
            actual_t = per_frame_cams[t]["actual_t"]
            s, R, tr = per_frame_transforms[t]
            print(f"── Processing NPZ/Rerun: t={t:02d} (frame={actual_t:06d}) ──")

            # GT pointcloud
            gt_pts = build_gt_pointcloud(actual_t, view_names, dataset_root, dataset_type=dataset_type)
            if gt_pts is None:
                print(f"  [WARN] No GT pointcloud at actual_t={actual_t}; skipping.")
                continue

            # Apply Umeyama to raw merged pointcloud
            raw_pts = per_frame_raw_pts[t]
            if len(raw_pts) > 0:
                aligned_pts = apply_similarity_transform(raw_pts, s, R, tr)
            else:
                aligned_pts = np.zeros((0, 3), dtype=np.float32)

            # ── Rerun logging ────────────────────────────────────────────
            if not no_rerun:
                rr.set_time("frame", sequence=actual_t)

            # Log GT cameras
            if not no_rerun:
                try:
                    log_cameras_rerun(actual_t, view_names, dataset_root, log_root, dataset_type=dataset_type)
                except Exception:
                    pass

            # Log Estimated cameras (aligned)
            if not no_rerun:
                try:
                    log_estimated_cameras_rerun(
                        actual_t, view_names,
                        per_frame_cams[t]["cam2world"],
                        per_frame_cams[t]["intrinsic"],
                        s, R, tr, log_root
                    )
                except Exception:
                    pass

            # GT (green) — subsample for Rerun stability
            if not no_rerun:
                try:
                    _sub = max(1, len(gt_pts) // 50000)
                    log_pointcloud(actual_t, f"{log_root}/gt", gt_pts[::_sub], color=[0, 255, 0])
                except Exception:
                    pass

            # GT Static (orange)
            if not no_rerun:
                try:
                    gt_static_pts = build_static_gt_pointcloud(actual_t, view_names, dataset_root, use_sam2=True,
                                                               dataset_type=dataset_type)
                    if gt_static_pts is not None and len(gt_static_pts) > 0:
                        _sub_static = max(1, len(gt_static_pts) // 50000)
                        log_pointcloud(t, f"{log_root}/gt_static", gt_static_pts[::_sub_static], color=[255, 165, 0])
                except Exception:
                    pass

            # Raw unaligned (orange)
            if not no_rerun and len(raw_pts) > 0:
                try:
                    _sub = max(1, len(raw_pts) // 50000)
                    rr.log(
                        f"{log_root}/raw/pointcloud",
                        rr.Points3D(
                            positions=raw_pts[::_sub],
                            radii=0.002,
                            colors=np.array([255, 128, 0], dtype=np.uint8),
                        ),
                    )
                except Exception:
                    pass

            # Aligned (blue)
            if not no_rerun and len(aligned_pts) > 0:
                try:
                    _sub = max(1, len(aligned_pts) // 50000)
                    rr.log(
                        f"{log_root}/aligned/pointcloud",
                        rr.Points3D(
                            positions=aligned_pts[::_sub],
                            radii=0.002,
                            colors=np.array([0, 128, 255], dtype=np.uint8),
                        ),
                    )
                except Exception:
                    pass

            # ── Save NPZ ─────────────────────────────────────────────────
            # GT camera params
            valid_Ks, valid_R_ts = [], []
            for vname in view_names:
                view_dir = os.path.join(dataset_root, vname)
                K, cam2world_gt = load_gt_params(view_dir, dataset_type=dataset_type)
                valid_Ks.append(K)
                valid_R_ts.append(np.linalg.inv(cam2world_gt))

            save_dict = {
                "gt_pts": gt_pts,
                "aligned_pts": aligned_pts,
                "scale": float(s),
                "R": R,
                "tr": tr,
                "pointmaps": per_frame_pointmaps[t],  # (V, H, W, 3)
                "pointmaps_confs": per_frame_confs[t],  # (V, H, W)
                "masks_2d": per_frame_masks[t],  # (V, H, W) True=static
                "frame_idx": int(actual_t),
                "Ks": np.array(valid_Ks),
                "R_ts": np.array(valid_R_ts),
                "est_poses": per_frame_cams[t]["cam2world"],
                "est_intrinsics": per_frame_cams[t]["intrinsic"],
                "min_conf_thr": float(frame_thr),
                "conf_percentile": float(CONF_PERCENTILE),
            }
            np.savez(out_frame_path, **save_dict)
            frame_times_sec.append(time.perf_counter() - frame_start)

        except Exception as e:
            print(f"  [ERROR] Frame t={t} failed: {e}")
            traceback.print_exc()
            continue

    return frame_times_sec


# ═══════════════════════════════════════════════════════════════════════════════
# Full reconstruction pipeline
# ═══════════════════════════════════════════════════════════════════════════════

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
        all_at_once=False,  # kept for backward compat with CLI, ignored
        no_rerun=False,
        dataset_type="dex-ycb",
):
    # ── Rerun setup ──────────────────────────────────────────────────────
    rerun_stream = f"vggt4d_{run_tag}"
    if not skip_rerun_init and not no_rerun:
        try:
            rr.init(rerun_stream, spawn=False)
            rr.connect_grpc(RERUN_ADDR)
        except Exception as e:
            print(f"[WARN] Rerun init failed for {run_tag}: {e}")

    from vggt.utils.rerun_logging import configure_rerun_view_defaults
    log_root = f"{run_tag}"
    if not skip_rerun_init and not no_rerun:
        rr.log(log_root, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        configure_rerun_view_defaults(log_root, RERUN_EYE_UP)

    # ── Discover views & frames ──────────────────────────────────────────
    from vggt.utils.camera_utils import build_views
    views = build_views(dataset_root, target_views=target_views, dataset_type=dataset_type)
    view_names = sorted(views.keys())
    if not view_names:
        print(f"[WARN] No valid views found for target_views={target_views}")
        return
    print(f"[INFO] Using views: {view_names}")

    os.makedirs(out_dir, exist_ok=True)
    run_start = time.perf_counter()

    # ── Process all frames at once ───────────────────────────────────────
    V = len(view_names)
    n_frames = len(views[view_names[0]])
    print(f"\\n[PHASE 1+2]  All-at-once pipeline  V={V}  T={n_frames}")

    frame_times_sec = run_all_at_once_pipeline(
        model, views, dataset_root, out_dir, run_tag, log_root,
        skip_existing_frames=skip_existing_frames,
        no_rerun=no_rerun,
        dataset_type=dataset_type,
    )

    # ── Timing report ────────────────────────────────────────────────────
    total_sec = time.perf_counter() - run_start
    timing_payload = {
        "strategy": os.path.basename(out_dir),
        "n_frames": int(n_frames),
        "total_seconds": float(total_sec),
        "seconds_per_frame": float(total_sec / max(len(frame_times_sec), 1)),
    }
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, indent=2)
    print(f"\n[TIME] total={total_sec:.2f}s  per_frame={timing_payload['seconds_per_frame']:.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=["dex-ycb", "hi4d"], default="dex-ycb", help="Dataset to use")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--subjects", nargs="+", type=str, help="Specific subject codes to run.")
    parser.add_argument("--views", nargs="+", type=int)
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--all-at-once", action="store_true", help="Process entire sequence in one batch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    parser.add_argument("--method", type=str, default="baseline", help="Method name (subfolder in aligned_outputs)")
    args = parser.parse_args()

    dataset_type = args.data
    subj_map = get_subject_by_code(dataset_type)
    dataset_config = get_dataset_config(dataset_type)

    if args.all:
        selected_codes = list(subj_map.keys())
    elif args.subjects:
        selected_codes = args.subjects
    else:
        # Legacy flag check
        import sys
        selected_codes = [a.lstrip('-') for a in sys.argv if a.startswith('--') and a.lstrip('-') in subj_map]
        if not selected_codes:
            print(f"[WARN] No subject selection provided; defaulting to first subject.")
            selected_codes = [list(subj_map.keys())[0]]

    view_counts = args.views if args.views else ([2, 3, 4] if RUN_MULTI_VIEW_EVAL else [4])

    # Load VGGT4D model
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[INFO] Loading VGGT4D from '{VGGT4D_CHECKPOINT}' on {DEVICE} ...")
    model = VGGTFor4D()
    model.load_state_dict(torch.load(VGGT4D_CHECKPOINT, weights_only=True))
    model.eval()
    model = model.to(DEVICE)

    cache_root = os.path.join(tempfile.gettempdir(), "vggt4d_alignment_cache")
    os.makedirs(cache_root, exist_ok=True)

    for scode in selected_codes:
        subject_full = subj_map.get(scode)
        if not subject_full:
            print(f"[WARN] Subject code {scode} not found for dataset {dataset_type}")
            continue

        dataset_root = os.path.join(dataset_config["root"], subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found: {dataset_root}")
            continue

        for nviews in view_counts:
            safe_code = scode.replace("/", "_")
            view_root = f"vggt4d_{dataset_type}_{safe_code}_{nviews}views"
            baseline_dir = os.path.join("aligned_outputs", args.method, dataset_type, subject_full, f"{nviews}views")

            pair_name = subject_full.split("/")[0] if dataset_type == "hi4d" else None
            target_views = get_view_config(dataset_type, nviews, pair_name=pair_name)

            print(f"\n[RUN] Subject={scode} views={nviews} -> {baseline_dir}")
            run_reconstruction(
                model=model,
                dataset_root=dataset_root,
                target_views=target_views,
                out_dir=baseline_dir,
                cache_root=cache_root,
                run_tag=view_root,
                no_rerun=args.no_rerun,
                all_at_once=args.all_at_once,
                skip_existing_frames=not args.overwrite,
                dataset_type=dataset_type,
            )


if __name__ == "__main__":
    main()
