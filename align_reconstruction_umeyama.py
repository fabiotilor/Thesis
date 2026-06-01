import os
import tempfile
import argparse
import json
import time
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
from mast3r.utils.camera_utils import get_rgb_path, remove_outliers
from mast3r.utils.optical_flow import compute_static_mask

# Building Ground Truth
from mast3r.utils.gt import (
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
    DATASETS, DATASET_BASE_ROOT,
    MODEL_NAME, IMAGE_SIZE, DEVICE, RERUN_ADDR,
    VIEW_CONFIGS, DEFAULT_TARGET_VIEWS, SCENE_GRAPH, RERUN_EYE_UP,
    CONF_PERCENTILE, CLEAN_DEPTH
)

# NOTE: Import rerun logging lazily inside `run_reconstruction` to avoid
# circular-import issues when other modules import this file.
RUN_MULTI_VIEW_EVAL = True


# ── helpers ───────────────────────────────────────────────────────────────────
def get_masked_image(t, vname, rgb_path, cache_dir, dataset_root, mask_subjects=False, dataset_type="dex-ycb"):
    if not mask_subjects or dataset_type != "hi4d":
        return rgb_path

    # Verify that rgb_path corresponds to the correct frame
    rgb_filename = os.path.basename(rgb_path)
    expected_frame = f"{t:06d}"
    if expected_frame not in rgb_filename:
        print(f"  [WARN] Frame mismatch! Expected frame {t} but rgb_path is {rgb_filename}")

    # Lazy import to avoid circular dependencies
    from mast3r.utils.gt import _load_hi4d_seg_mask
    print(f"  [DEBUG] Loading mask for {vname} at frame {t} (from RGB: {rgb_filename})")
    mask = _load_hi4d_seg_mask(dataset_root, vname, t)
    if mask is None:
        return rgb_path

    # Apply mask and save to cache
    masked_dir = os.path.join(cache_dir, "masked_images")
    os.makedirs(masked_dir, exist_ok=True)
    out_path = os.path.join(masked_dir, f"{vname}_{t:06d}.jpg")

    if os.path.exists(out_path):
        return out_path

    img = cv2.imread(rgb_path)
    if img is None: return rgb_path

    H, W = img.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    # Debug: Save mask visualization and stats
    mask_coverage = 100 * np.sum(mask) / (H * W)
    print(f"  [DEBUG] {vname} mask coverage: {mask_coverage:.1f}% ({np.sum(mask)}/{H * W} pixels)")

    # Save mask as visualization
    mask_viz_path = os.path.join(masked_dir, f"{vname}_{t:06d}_mask.png")
    cv2.imwrite(mask_viz_path, mask.astype(np.uint8) * 255)

    img[~mask] = 0  # Black out background
    cv2.imwrite(out_path, img)

    # Debug: Save side-by-side comparison
    comparison_path = os.path.join(masked_dir, f"{vname}_{t:06d}_comparison.jpg")
    img_resized = cv2.resize(img, (W // 2, H // 2))
    original_resized = cv2.resize(cv2.imread(rgb_path), (W // 2, H // 2))
    comparison = np.hstack([original_resized, img_resized])
    cv2.imwrite(comparison_path, comparison)

    print(f"  [DEBUG] Applied mask for frame {t} to {vname}: {out_path} (source: {rgb_filename})")
    print(f"  [DEBUG] Saved mask visualization: {mask_viz_path}")
    print(f"  [DEBUG] Saved comparison: {comparison_path}")

    return out_path


def build_views(dataset_root, target_views=None, dataset_type="dex-ycb", start=0, step=1, limit=None):
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
                # Apply slicing
                sliced = frames[start::step]
                if limit:
                    sliced = sliced[:limit]
                views[vname] = sliced
    elif dataset_type == "hi4d":
        img_dir = os.path.join(dataset_root, "images")
        cam_ids = [str(v) for v in target_views] if target_views else sorted(os.listdir(img_dir))
        for cid in cam_ids:
            cid_dir = os.path.join(img_dir, cid)
            if not os.path.isdir(cid_dir): continue
            frames = sorted(f for f in glob.glob(os.path.join(cid_dir, "*"))
                            if os.path.splitext(f.lower())[1] in img_exts)
            if frames:
                # Apply slicing
                sliced = frames[start::step]
                if limit:
                    sliced = sliced[:limit]
                views[cid] = sliced
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
        subject_name=None,  # Added
        num_views=None,  # Added
        run_tag="default",
        skip_rerun_init=False,
        skip_existing_frames=True,
        no_rerun=False,
        dataset_type="dex-ycb",
        start=0,
        step=1,
        limit=None,
        mask_subjects=False,
):
    # Determine subject and views if not provided
    if subject_name is None:
        subject_name = os.path.basename(dataset_root)
    if num_views is None:
        num_views = len(target_views) if target_views else 0

    print(f"[INFO] Using adaptive percentile threshold: {CONF_PERCENTILE} for {subject_name} ({num_views} views)")
    rerun_stream = f"mast3r_stabilisation_{run_tag}"
    if not skip_rerun_init and not no_rerun:
        try:
            rr.init(rerun_stream, spawn=False)
            rr.connect_grpc(RERUN_ADDR)
        except Exception as e:
            print(f"[WARN] Rerun init failed for {run_tag}: {e}")

    # Lazy import to avoid circular imports.
    from mast3r.utils.rerun_logging import (
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
        configure_rerun_view_defaults(log_root, eye_up)

    views = build_views(dataset_root, target_views=target_views, dataset_type=dataset_type, start=start, step=step,
                        limit=limit)
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
    mask_base_dir = os.path.join(out_dir, "flow_masks")

    for t in range(n_frames):
        # Extract frame index if possible (for Hi4D it might not be simple t)
        first_view_path = views[view_names[0]][t]
        frame_filename = os.path.basename(first_view_path)
        try:
            actual_t = int(os.path.splitext(frame_filename)[0])
        except ValueError:
            actual_t = t

        out_frame_path = os.path.join(out_dir, f"frame_{actual_t:06d}.npz")
        if skip_existing_frames and os.path.exists(out_frame_path):
            print(f"  [SKIP] {run_tag}: existing {os.path.basename(out_frame_path)} found.")
            continue

        frame_start = time.perf_counter()
        try:
            print(f"── t={actual_t:06d} / {n_frames - 1} ──────────────────────────────────────")

            # Use the correct frame paths for this iteration
            current_frame_paths = [views[v][t] for v in view_names]

            hi4d_masks = []
            masked_current_files = []
            for i, v in enumerate(view_names):
                m_img = get_masked_image(actual_t, v, current_frame_paths[i], cache_root, dataset_root,
                                         mask_subjects=mask_subjects, dataset_type=dataset_type)
                masked_current_files.append(m_img)

                # Cache the mask for output filtering
                if dataset_type == "hi4d":
                    hi4d_masks.append(_load_hi4d_seg_mask(dataset_root, v, actual_t))
                else:
                    hi4d_masks.append(None)

            if not no_rerun:
                # Create mapping of view names to masked image paths for logging
                masked_image_paths = {}
                for i, v in enumerate(view_names):
                    if mask_subjects and dataset_type == "hi4d":
                        # Only use masked images for logging if they're different from original
                        original_path = views[v][i]
                        masked_path = masked_current_files[i]
                        if masked_path != original_path:
                            masked_image_paths[v] = masked_path

                log_cameras_rerun(actual_t, view_names, dataset_root, log_root,
                                  dataset_type=dataset_type, masked_image_paths=masked_image_paths)

            imgs = load_images(masked_current_files, size=IMAGE_SIZE)
            pairs = make_pairs(imgs, scene_graph=SCENE_GRAPH, symmetrize=True)
            scene = sparse_global_alignment(
                masked_current_files,
                pairs,
                os.path.join(run_cache_root, f"t{t:02d}"),
                model,
                device=DEVICE,
                matching_conf_thr=0.0,
                niter1=300, niter2=300
            )

            pts3d_list, depthmaps, confs = to_numpy(
                scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH and not mask_subjects))

            # Apply segmentation masks to output pointmaps for hi4d
            # This zeros out background points that the model may still output
            if dataset_type == "hi4d":
                for i, vname in enumerate(view_names):
                    seg_mask = hi4d_masks[i]
                    if seg_mask is not None:
                        H, W = confs[i].shape[:2]
                        # Resize mask to match model output resolution
                        mask_resized = cv2.resize(
                            seg_mask.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        # Zero out background points in both pointmap and confidence
                        mask_flat = mask_resized.ravel()
                        pts3d_list[i].reshape(-1, 3)[~mask_flat] = 0
                        confs[i].reshape(-1)[~mask_flat] = 0

            # Debug: Log point counts before filtering
            total_points_before = sum(len(pts.ravel()) for pts in pts3d_list)
            print(f"  [DEBUG] Total points before filtering: {total_points_before}")

            # ── Precompute Flow Masks ──────────────────────────────────────────
            precomputed_masks = {}
            for i, vname in enumerate(view_names):
                view_dir_v = os.path.join(dataset_root, "images", vname) if dataset_type == "hi4d" else os.path.join(
                    dataset_root, vname)
                # Use the current frame path instead of trying to construct it
                rgb_t_v = current_frame_paths[i]
                # Try to get adjacent frames for flow computation
                if t + 1 < n_frames:
                    rgb_adj_v = views[vname][t + 1]
                elif t - 1 >= 0:
                    rgb_adj_v = views[vname][t - 1]
                else:
                    rgb_adj_v = None

                rgb_paths_v = [p for p in [rgb_t_v, rgb_adj_v] if p is not None]
                if len(rgb_paths_v) >= 2:
                    precomputed_masks[vname] = compute_static_mask(rgb_paths_v)
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
                print(f"  [WARN] No GT pointcloud at t={t}; skipping frame.")
                continue

            # ── 3. Similarity Alignment to GT ──────────────────────────────────
            # Estimate similarity transform using ALL correspondences
            src_points, dst_points = get_static_correspondences(
                actual_t, view_names, scene, dataset_root,
                precomputed_masks=precomputed_masks,
                use_static_mask=False,
                dataset_type=dataset_type
            )
            s, R, tr = estimate_similarity_transform(src_points, dst_points)

            # ── Filter and Align estimated points ──────────────────────────────
            est_pts_parts = []
            total_points_after_filtering = 0

            # For Hi4D, skip GT validity mask since masked images already ensure person-only pixels
            use_gt_mask = dataset_type != "hi4d"

            if use_gt_mask:
                dataset_cfg = DATASETS.get(dataset_type, {})
                depth_max = dataset_cfg.get("depth_max_m", DEPTH_MAX_M)
                gt_validity_masks = build_gt_validity_masks(
                    actual_t, view_names, dataset_root,
                    depth_max_m=depth_max if depth_max is not None else 999.0,
                    target_hw=None,
                    dataset_type=dataset_type
                )

            for i, vname in enumerate(view_names):
                pts_i = pts3d_list[i].reshape(-1, 3)
                conf_i = confs[i].ravel()
                thr_i = np.percentile(conf_i, 100 * (1 - CONF_PERCENTILE))
                conf_ok = conf_i > thr_i

                if use_gt_mask:
                    gt_mask = gt_validity_masks[i]
                    if gt_mask is None: continue

                    H, W = confs[i].shape[:2]
                    if gt_mask.shape != (H, W):
                        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(
                            bool)

                    valid = conf_ok & gt_mask.ravel()
                else:
                    valid = conf_ok

                raw_pts = pts_i[valid]
                clean_pts = remove_outliers(raw_pts, nb_neighbors=15, std_ratio=0.8)
                total_points_after_filtering += len(clean_pts)
                est_pts_parts.append(clean_pts)

            print(
                f"  [DEBUG] Total points after filtering: {total_points_after_filtering} ({100 * total_points_after_filtering / total_points_before:.1f}% retained)")

            if est_pts_parts:
                est_pts = np.concatenate(est_pts_parts, axis=0)
                aligned_pts = apply_similarity_transform(est_pts, s, R, tr)
                print(f"  [DEBUG] Final aligned points: {len(aligned_pts)}")
            else:
                print(f"  [WARN] No aligned points for t={actual_t}")
                continue

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

            try:
                im_poses = scene.get_im_poses()
            except AttributeError:
                im_poses = scene.get_poses()
            est_poses_all = to_numpy(im_poses)

            # FIX: intrinsics is a method call, not a plain attribute
            try:
                est_intrinsics_all = to_numpy(scene.get_intrinsics())
            except AttributeError:
                est_intrinsics_all = to_numpy(scene.intrinsics)

            # ── Per-view flow masks ────────────────────────────────────────────
            for i, vname in enumerate(view_names):
                if dataset_type == "hi4d":
                    view_dir = os.path.join(dataset_root, "images", vname)
                else:
                    view_dir = os.path.join(dataset_root, vname)

                K, cam2world = load_gt_params(view_dir, dataset_type=dataset_type)
                R_t = np.linalg.inv(cam2world)

                flow_mask = precomputed_masks.get(vname)
                if flow_mask is None:
                    print(f"  [WARN] {vname} t={t}: not enough frames or flow mask failed, skipping view")
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
                    os.path.join(view_mask_out, f"flow_mask_{t:02d}.png"),
                    flow_mask_mod.astype(np.uint8) * 255,
                )

                if use_gt_mask:
                    if gt_validity_masks is not None and i < len(gt_validity_masks):
                        gt_mask = gt_validity_masks[i]
                        if gt_mask is not None:
                            if gt_mask.shape != (H_mod, W_mod):
                                gt_mask = cv2.resize(
                                    gt_mask.astype(np.uint8), (W_mod, H_mod),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(bool)
                            cv2.imwrite(
                                os.path.join(view_mask_out, f"gt_mask_{t:02d}.png"),
                                gt_mask.astype(np.uint8) * 255,
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
                'view_names': np.array(view_names)  # Save view names for alignment strategies
            }
            if valid_masks:
                save_dict['masks_2d'] = np.stack(valid_masks)

            np.savez(out_frame_path, **save_dict)
            frame_times_sec.append(time.perf_counter() - frame_start)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] Frame actual_t={actual_t} failed: {e}")
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


def main():
    from mast3r.utils.rerun_logging import add_dataset_args, get_selected_subjects
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    parser.add_argument(
        "--views",
        nargs="+",
        type=int,
        help="Optional view counts to run (e.g. --views 2 3 4).",
    )
    args = parser.parse_args()

    selected_subjects, codes = get_selected_subjects(args)
    dataset_type = args.data
    dataset_cfg = DATASETS[dataset_type]

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[INFO] Dataset: {dataset_type}")
    print(f"[INFO] loading model '{MODEL_NAME}' …")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    cache_root = os.path.join(tempfile.gettempdir(), f"mast3r_alignment_cache_{dataset_type}")
    os.makedirs(cache_root, exist_ok=True)

    print(f"[INFO] Selected subjects: {codes}")
    rr.init("mast3r_stabilisation", spawn=False)
    rr.connect_grpc(RERUN_ADDR)
    for subject_name in selected_subjects:
        dataset_root = os.path.join(dataset_cfg["root"], subject_name)
        if not os.path.isdir(dataset_root):
            print(f"[WARN] Subject directory not found, skipping: {dataset_root}")
            continue

        print(f"\n[INFO] Processing subject: {subject_name}")

        # Determine view counts to run
        if args.views:
            camera_counts = args.views
        else:
            view_configs = dataset_cfg.get("view_configs", {})
            # If hi4d, use the keys of view_configs[subject_prefix] or default
            if dataset_type == "hi4d":
                pair_prefix = subject_name.split("/")[0]
                cfg = view_configs.get(pair_prefix, view_configs.get("default", {}))
                camera_counts = sorted([int(k) for k in cfg.keys()])
            else:
                camera_counts = [2, 3, 4] if RUN_MULTI_VIEW_EVAL else [4]

        for num_views in camera_counts:
            # Resolve target views
            view_configs = dataset_cfg.get("view_configs", {})
            if dataset_type == "hi4d":
                pair_prefix = subject_name.split("/")[0]
                cfg = view_configs.get(pair_prefix, view_configs.get("default", {}))
                target_views = cfg.get(num_views)
            else:
                target_views = view_configs.get(num_views)

            if target_views is None:
                target_views = dataset_cfg.get("default_target_views")

            out_dir = os.path.join("aligned_outputs", dataset_type, "baseline", subject_name, f"{num_views}views")
            run_reconstruction(
                model=model,
                dataset_root=dataset_root,
                target_views=target_views,
                out_dir=out_dir,
                cache_root=cache_root,
                subject_name=subject_name,
                num_views=num_views,
                run_tag=f"{dataset_type}_{subject_name}_{num_views}views",
                dataset_type=dataset_type,
                start=args.start,
                step=args.step,
                limit=args.limit,
                mask_subjects=args.mask_subjects,
            )

    print("[done]")
    print("\nRun metrics with:")
    print("  python evaluate_temporal_consistency.py")


if __name__ == "__main__":
    main()