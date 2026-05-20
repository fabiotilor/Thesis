import os
import argparse
import tempfile
import torch
import numpy as np
import cv2
import glob

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import rerun as rr
except ImportError:
    rr = None

from eval_config import (
    CONF_PERCENTILE,
    DATASET_BASE_ROOT,
    SUBJECT_BY_CODE,
    VGGT4D_CHECKPOINT,
    DEVICE,
    IMAGE_SIZE,
    RERUN_ADDR,
    RERUN_EYE_UP,
    DATASETS,
    get_dataset_config,
    get_subject_by_code,
    get_view_config
)

from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.utils.model_utils import run_vggt4d_3stage_inference
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.gt import load_gt_params, build_gt_validity_masks, build_gt_pointcloud, DEPTH_MAX_M, DEPTH_SCALE

from vggt.utils.camera_utils import build_views


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT4D-to-GGPT Data Preparation Script")
    parser.add_argument("--data", type=str, choices=["dex-ycb", "hi4d"], default="dex-ycb", help="Dataset to use")
    parser.add_argument("--subjects", nargs="+", type=str, help="Specific subject codes to run (e.g. pair00/dance00)")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--views", nargs="+", type=int, default=None, help="Number of views to use (e.g. 2 3 4)")
    parser.add_argument("--output_dir", type=str, default="ggpt_inputs", help="Root output directory")
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun logging")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_type = args.data
    dataset_config = get_dataset_config(dataset_type)
    subj_map = get_subject_by_code(dataset_type)

    if args.all:
        subjects_to_process = sorted(subj_map.keys())
        print(f"[INFO] Processing all subjects: {subjects_to_process}")
    elif args.subjects:
        subjects_to_process = args.subjects
    else:
        print(f"[WARN] No subject selection provided; defaulting to first subject.")
        subjects_to_process = [list(subj_map.keys())[0]]

    view_counts = args.views if args.views is not None else [2, 3, 4]

    print(f"[INFO] Loading VGGT4D model on {DEVICE}...")
    model = VGGTFor4D()
    model.load_state_dict(torch.load(VGGT4D_CHECKPOINT, weights_only=True))
    model.eval().to(DEVICE)

    for scode in subjects_to_process:
        subject_full = subj_map.get(scode, scode)

        # Check if subject already processed
        subject_out_dir = os.path.join(args.output_dir, f"subject-{scode}")
        if os.path.isdir(subject_out_dir):
            print(f"[SKIP] Subject {scode} already exists at {subject_out_dir}")
            continue

        dataset_root = os.path.join(dataset_config["root"], subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[ERROR] Subject directory not found: {dataset_root}")
            continue

        print(f"\n" + "=" * 50)
        print(f"PROCESSING SUBJECT: {scode} ({subject_full})")
        print("=" * 50)

        if not args.no_rerun and rr is not None:
            from vggt.utils.rerun_logging import initialize_rerun_session
            initialize_rerun_session(f"ggpt_prep_vggt4d_{scode}", RERUN_ADDR, log_root="world")

        for n_views in view_counts:
            print(f"\n[INFO] Views = {n_views}")
            views_out_dir = os.path.join(subject_out_dir, f"{n_views}views")
            os.makedirs(views_out_dir, exist_ok=True)

            pair_name = subject_full.split("/")[0] if dataset_type == "hi4d" else None
            target_view_names = get_view_config(dataset_type, n_views, pair_name=pair_name)
            if not target_view_names:
                print(f"[WARN] Unknown view config for dataset {dataset_type} / views {n_views}")
                continue

            views_dict = build_views(dataset_root, target_views=target_view_names, dataset_type=dataset_type)
            view_names = sorted(views_dict.keys())
            if not view_names:
                print(f"[ERROR] No views found for target count {n_views}")
                continue

            n_frames = len(views_dict[view_names[0]])
            V = len(view_names)
            print(f"[INFO] Using views: {view_names}")

            # ── Inference over entire sequence for VGGT4D ──
            seq_paths = []
            for t in range(n_frames):
                for vname in view_names:
                    seq_paths.append(views_dict[vname][t])

            # ── Pre-mask HI4D input images (matches align_reconstruction_umeyama.py) ──
            if dataset_type == "hi4d":
                from vggt.utils.gt import _load_hi4d_seg_mask
                print("[INFO] Pre-masking HI4D input images...")
                masked_seq_paths = []
                cache_dir = os.path.join(tempfile.gettempdir(), "vggt4d_hi4d_masked")
                os.makedirs(cache_dir, exist_ok=True)

                for i, fpath in enumerate(seq_paths):
                    t_idx = i // V
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

            print(f"[INFO] Running VGGT4D inference on {len(seq_paths)} images...")
            chunk = run_vggt4d_3stage_inference(model, seq_paths, DEVICE)

            world_points = chunk["world_points"]  # (T*V, H, W, 3)
            world_confs = chunk["world_points_conf"]  # (T*V, H, W)
            extrinsic = chunk["extrinsic"]  # (T*V, 3, 4)
            intrinsic = chunk["intrinsic"]  # (T*V, 3, 3)

            all_points = []
            all_confs = []
            all_images = []
            all_extrinsics = []
            all_intrinsics = []
            all_gt_points = []
            all_gt_masks = []
            all_gt_extrinsics = []
            all_gt_intrinsics = []
            all_point_masks = []

            for t in tqdm(range(n_frames), desc=f"Frames ({scode} - {n_views}v)"):
                current_files = [views_dict[v][t] for v in view_names]
                # Dynamically extract actual_t from filename (handles arbitrary sequence start offsets)
                first_view_path = current_files[0]
                actual_t = int(os.path.splitext(os.path.basename(first_view_path))[0])
                imgs = load_and_preprocess_images(current_files)  # [V, 3, H, W], normalized [0, 1]

                idx_start = t * V
                idx_end = idx_start + V

                pts3d_all = world_points[idx_start:idx_end]
                confs_all = world_confs[idx_start:idx_end]
                extrinsics_est = extrinsic[idx_start:idx_end]
                intrinsics_est = intrinsic[idx_start:idx_end]

                V_count, H_mod, W_mod = confs_all.shape

                # ── Post-mask output pointmaps with crop-aware seg masks (matches align_reconstruction_umeyama.py) ──
                if dataset_type == "hi4d":
                    from vggt.utils.gt import _load_hi4d_seg_mask, _resize_mask_crop_aware
                    for i_v, vname in enumerate(view_names):
                        seg_mask = _load_hi4d_seg_mask(dataset_root, vname, actual_t)
                        if seg_mask is not None:
                            H_orig, W_orig = seg_mask.shape[:2]
                            mask_resized = _resize_mask_crop_aware(
                                seg_mask, W_orig, H_orig, H_mod, W_mod
                            )
                            pts3d_all[i_v][~mask_resized] = 0
                            confs_all[i_v][~mask_resized] = 0

                # ── Global confidence threshold for this frame ──
                all_confs_f = np.concatenate([c.ravel() for c in confs_all])
                frame_thr = np.quantile(all_confs_f, 1.0 - CONF_PERCENTILE)
                min_conf_thresh = 0.01 if dataset_type == "hi4d" else 0.0

                gt_validity_masks = build_gt_validity_masks(actual_t, view_names, dataset_root,
                                                            target_hw=(H_mod, W_mod), depth_max_m=DEPTH_MAX_M,
                                                            dataset_type=dataset_type)

                for i in range(V_count):
                    pts_i = pts3d_all[i]
                    conf_i = confs_all[i]

                    conf_mask = (conf_i > frame_thr) & (conf_i > min_conf_thresh)

                    gt_mask = gt_validity_masks[i]
                    if gt_mask is None:
                        gt_mask = np.ones((H_mod, W_mod), dtype=bool)
                    else:
                        if gt_mask.shape != (H_mod, W_mod):
                            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (W_mod, H_mod),
                                                 interpolation=cv2.INTER_NEAREST).astype(bool)

                    final_mask = conf_mask & gt_mask
                    pts_i_masked = pts_i.copy()
                    pts_i_masked[~final_mask] = 0
                    conf_i_masked = conf_i.copy()
                    conf_i_masked[~final_mask] = 0

                    all_points.append(pts_i_masked)
                    all_confs.append(conf_i_masked)

                    img_i = imgs[i].permute(1, 2, 0).cpu().float().numpy()
                    all_images.append(img_i)

                    # Extrinsics: VGGT4D outputs extrinsic directly (3x4), we pad it to 4x4
                    ext_4x4 = np.eye(4)
                    ext_4x4[:3, :4] = extrinsics_est[i]
                    all_extrinsics.append(ext_4x4)
                    all_intrinsics.append(intrinsics_est[i])
                    all_point_masks.append(final_mask)

                    # GT Data
                    view_dir = os.path.join(dataset_root, view_names[i])
                    K_gt, cam2world_gt = load_gt_params(view_dir, dataset_type=dataset_type)
                    w2c_gt = np.linalg.inv(cam2world_gt)
                    all_gt_extrinsics.append(w2c_gt)

                    depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
                    if os.path.exists(depth_path):
                        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                        depth_m = depth_raw * DEPTH_SCALE
                        H_orig, W_orig = depth_raw.shape[:2]
                        K_small = K_gt.copy()
                        K_small[0, 0] *= (W_mod / W_orig)
                        K_small[1, 1] *= (H_mod / H_orig)
                        K_small[0, 2] *= (W_mod / W_orig)
                        K_small[1, 2] *= (H_mod / H_orig)
                        all_gt_intrinsics.append(K_small)

                        depth_small = cv2.resize(depth_m, (W_mod, H_mod), interpolation=cv2.INTER_NEAREST)
                        fy, fx = K_small[1, 1], K_small[0, 0]
                        cy, cx = K_small[1, 2], K_small[0, 2]
                        v_grid, u_grid = np.meshgrid(np.arange(H_mod), np.arange(W_mod), indexing='ij')
                        pts_cam_gt = np.stack(
                            [(u_grid - cx) * depth_small / fx, (v_grid - cy) * depth_small / fy, depth_small],
                            axis=-1)
                        pts_world_gt = (cam2world_gt[:3, :3] @ pts_cam_gt.reshape(-1, 3).T).T + cam2world_gt[:3, 3]
                        pts_world_gt = pts_world_gt.reshape(H_mod, W_mod, 3)
                        gt_mask_small = (depth_small > 0) & (depth_small <= DEPTH_MAX_M)
                        pts_world_gt[~gt_mask_small] = 0
                        all_gt_points.append(pts_world_gt)
                        all_gt_masks.append(gt_mask_small)
                    else:
                        all_gt_points.append(np.zeros((H_mod, W_mod, 3), dtype=np.float32))
                        all_gt_masks.append(np.zeros((H_mod, W_mod), dtype=bool))
                        if dataset_type == "hi4d":
                            from vggt.utils.gt import _compute_crop_aware_transform
                            img_path = current_files[i]
                            if os.path.exists(img_path):
                                tmp_img = cv2.imread(img_path)
                                H_orig, W_orig = tmp_img.shape[:2]
                            else:
                                H_orig, W_orig = 1280, 940
                            scale_x, scale_y, crop_y_offset = _compute_crop_aware_transform(W_orig, H_orig, H_mod,
                                                                                            W_mod)
                            K_small = np.array([
                                [K_gt[0, 0] * scale_x, 0, K_gt[0, 2] * scale_x],
                                [0, K_gt[1, 1] * scale_y, K_gt[1, 2] * scale_y - crop_y_offset],
                                [0, 0, 1]
                            ])
                        else:
                            K_small = K_gt.copy()
                            K_small[0, 0] *= (W_mod / 640)
                            K_small[1, 1] *= (H_mod / 480)
                            K_small[0, 2] *= (W_mod / 640)
                            K_small[1, 2] *= (H_mod / 480)
                        all_gt_intrinsics.append(K_small)

                # ── Rerun Logging ────────────────────────────────────────────────
                if not args.no_rerun and rr is not None:
                    rr.set_time("frame", sequence=actual_t)
                    f_m_pts = []
                    for i in range(-V_count, 0):
                        m_pts = all_points[i]
                        m_mask = all_point_masks[i]
                        if m_mask.any(): f_m_pts.append(m_pts[m_mask])
                    if f_m_pts:
                        rr.log(f"world/subject_{scode}/points",
                               rr.Points3D(np.concatenate(f_m_pts), colors=[255, 200, 0], radii=0.002))

                    gt_cloud = build_gt_pointcloud(actual_t, view_names, dataset_root, dataset_type=dataset_type)
                    if gt_cloud is not None:
                        rr.log(f"world/subject_{scode}/gt_points",
                               rr.Points3D(gt_cloud, colors=[0, 255, 100], radii=0.002))

            # ── Save Bins for this view count ──────────────────────────────────
            out_path = os.path.join(args.output_dir, f"subject-{scode}", f"{n_views}views")
            os.makedirs(out_path, exist_ok=True)
            print(f"[INFO] Saving results to {out_path}...")

            torch.save({
                "points": torch.from_numpy(np.stack(all_points)),
                "points_conf": torch.from_numpy(np.stack(all_confs)),
                "images_ff": torch.from_numpy(np.stack(all_images)),
                "extrinsics": torch.from_numpy(np.stack(all_extrinsics)),
                "intrinsics": torch.from_numpy(np.stack(all_intrinsics)),
            }, os.path.join(out_path, "ff_outputs.bin"))

            torch.save({
                "points": torch.from_numpy(np.stack(all_points)),
                "point_masks": torch.from_numpy(np.stack(all_point_masks)),
            }, os.path.join(out_path, "sfm_dlt_outputs.bin"))

            torch.save({
                "points": torch.from_numpy(np.stack(all_gt_points)),
                "point_masks": torch.from_numpy(np.stack(all_gt_masks)),
                "extrinsics": torch.from_numpy(np.stack(all_gt_extrinsics)),
                "intrinsics": torch.from_numpy(np.stack(all_gt_intrinsics)),
            }, os.path.join(out_path, "gt.bin"))

        print(f"[SUCCESS] Export complete for subject {scode}.")


if __name__ == "__main__":
    main()
