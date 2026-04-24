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
)

CLEAN_DEPTH = True
RUN_MULTI_VIEW_EVAL = True


def run_all_at_once_pipeline(
        model, views, dataset_root, out_dir, run_tag, log_root,
        skip_existing_frames=True,
        no_rerun=False,
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

    # ── 2. Run VGGT4D 3-stage inference ──────────────────────────────────
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

    # ── 4. Merge per-frame & collect correspondences ─────────────────────
    all_src, all_dst = [], []

    # Per-frame raw merged pointclouds and metadata (stored for Phase 2)
    per_frame_raw_pts = []  # list of (N_t, 3)
    per_frame_raw_confs = []  # list of (N_t,)
    per_frame_pointmaps = []  # (V, H, W, 3)  for NPZ backward compat
    per_frame_confs = []  # (V, H, W)     for NPZ
    per_frame_masks = []  # (V, H, W) bool
    per_frame_cams = []  # dict per frame

    for t in range(n_frames):
        idx_start = t * V
        idx_end = idx_start + V

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
        })

        # ── Compute Global Confidence Threshold for this Frame ─────────────
        all_confs_f = np.concatenate([c.ravel() for c in conf_views])
        frame_thr = np.quantile(all_confs_f, 1.0 - CONF_PERCENTILE)

        merged_pts_list = []
        merged_conf_list = []

        for vi, vname in enumerate(view_names):
            pts_flat = pm_views[vi].reshape(-1, 3)
            conf_flat = conf_views[vi].ravel()
            static = ~dyn_views[vi]

            # Collect correspondences (model pts → GT world pts)
            src, dst = get_single_view_correspondences(
                t, vname, pm_views[vi], conf_views[vi],
                dataset_root,
                static_mask=static,
                conf_percentile=CONF_PERCENTILE,
            )
            if src is not None and len(src) > 0:
                all_src.append(src)
                all_dst.append(dst)

            # Merge valid points
            valid = (conf_flat > frame_thr)
            gt_validity = build_gt_validity_masks(
                t, [vname], dataset_root,
                depth_max_m=DEPTH_MAX_M, target_hw=(H, W),
            )
            if gt_validity[0] is not None:
                gt_mask = gt_validity[0]
                if gt_mask.shape != (H, W):
                    gt_mask = cv2.resize(
                        gt_mask.astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                valid &= gt_mask.ravel()

            if valid.any():
                merged_pts_list.append(pts_flat[valid])
                merged_conf_list.append(conf_flat[valid])

        raw_merged = (
            np.concatenate(merged_pts_list) if merged_pts_list
            else np.zeros((0, 3), dtype=np.float32)
        )
        raw_merged_conf = (
            np.concatenate(merged_conf_list) if merged_conf_list
            else np.zeros((0,), dtype=np.float32)
        )
        per_frame_raw_pts.append(raw_merged)
        per_frame_raw_confs.append(raw_merged_conf)

    # Free GPU memory
    del chunk, world_points, world_confs, dyn_masks
    torch.cuda.empty_cache()
    gc.collect()

    # ── 5. Solve ONE global Umeyama from ALL correspondences ─────────────
    if all_src:
        src_cat = np.concatenate(all_src)
        dst_cat = np.concatenate(all_dst)
        s, R, tr = estimate_similarity_transform(src_cat, dst_cat)
        print(f"  Global Umeyama:  scale={s:.4f}  corr={len(src_cat):,}")
    else:
        s, R, tr = 1.0, np.eye(3), np.zeros(3)
        print("  Global Umeyama:  [WARN] no correspondences → identity")

    # ── 6. Apply alignment & save per-frame NPZ ─────────────────────────
    frame_times_sec = []
    for t in range(n_frames):
        out_frame_path = os.path.join(out_dir, f"frame_{t:02d}.npz")
        if skip_existing_frames and os.path.exists(out_frame_path):
            print(f"  [SKIP] {run_tag}: existing {os.path.basename(out_frame_path)}")
            continue

        frame_start = time.perf_counter()
        try:
            print(f"── t={t:02d} / {n_frames - 1} ──")

            # GT pointcloud
            gt_pts = build_gt_pointcloud(t, view_names, dataset_root)
            if gt_pts is None:
                print(f"  [WARN] No GT pointcloud at t={t}; skipping.")
                continue

            # Apply Umeyama to raw merged pointcloud
            raw_pts = per_frame_raw_pts[t]
            if len(raw_pts) > 0:
                aligned_pts = apply_similarity_transform(raw_pts, s, R, tr)
            else:
                aligned_pts = np.zeros((0, 3), dtype=np.float32)

            # ── Rerun logging ────────────────────────────────────────────
            if not no_rerun:
                rr.set_time("frame", sequence=t)

            # Log GT cameras
            if not no_rerun:
                try:
                    log_cameras_rerun(t, view_names, dataset_root, log_root)
                except Exception:
                    pass

            # GT (green) — subsample for Rerun stability
            if not no_rerun:
                try:
                    _sub = max(1, len(gt_pts) // 50000)
                    log_pointcloud(t, f"{log_root}/gt", gt_pts[::_sub], color=[0, 255, 0])
                except Exception:
                    pass

            # GT Static (orange)
            if not no_rerun:
                try:
                    gt_static_pts = build_static_gt_pointcloud(t, view_names, dataset_root, use_sam2=True)
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
                K, cam2world_gt = load_gt_params(view_dir)
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
                "frame_idx": int(t),
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
    views = build_views(dataset_root, target_views=target_views)
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
    parser.add_argument("--all", action="store_true")
    for code in SUBJECT_BY_CODE.keys():
        parser.add_argument(f"--{code}", action="store_true")
    parser.add_argument("--views", nargs="+", type=int)
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--all-at-once", action="store_true", help="Process entire sequence in one batch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    parser.add_argument("--method", type=str, default="baseline", help="Method name (subfolder in aligned_outputs)")
    args = parser.parse_args()

    selected_codes = [c for c in SUBJECT_BY_CODE.keys() if getattr(args, c)]
    if args.all:
        selected_codes = list(SUBJECT_BY_CODE.keys())
    if not selected_codes:
        print("[WARN] No subject selected, defaulting to --01")
        selected_codes = ["01"]

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
        subject_full = SUBJECT_BY_CODE[scode]
        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found: {dataset_root}")
            continue

        for nviews in view_counts:
            view_root = f"vggt4d_{scode}_{nviews}views"
            baseline_dir = os.path.join("aligned_outputs", args.method, subject_full, f"{nviews}views")

            target_views = VIEW_CONFIGS.get(nviews) or DEFAULT_TARGET_VIEWS

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
            )


if __name__ == "__main__":
    main()
