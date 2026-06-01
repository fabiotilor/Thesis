import os
import argparse
import torch
import numpy as np
import cv2
import glob
import tempfile
from tqdm import tqdm
from collections import defaultdict

try:
    import rerun as rr
except ImportError:
    rr = None

from eval_config import (
    CONF_PERCENTILE,
    DATASETS,
    MODEL_NAME,
    DEVICE,
    IMAGE_SIZE,
    RERUN_ADDR,
    RERUN_EYE_UP,
    VIEW_CONFIGS,
    DEFAULT_TARGET_VIEWS,
    SCENE_GRAPH,
    get_subject_by_code
)

# MASt3R / DUSt3R imports
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

from mast3r.utils.gt import load_gt_params, build_gt_validity_masks, build_gt_pointcloud, DEPTH_MAX_M, DEPTH_SCALE, \
    _load_hi4d_seg_mask
from mast3r.utils.rerun_logging import add_dataset_args, get_selected_subjects

CLEAN_DEPTH = True


def get_masked_image(t, vname, rgb_path, cache_dir, dataset_root, mask_subjects=False, dataset_type="dex-ycb"):
    if not mask_subjects or dataset_type != "hi4d":
        return rgb_path

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

    img[~mask] = 0  # Black out background
    cv2.imwrite(out_path, img)
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
                sliced = frames[start::step]
                if limit: sliced = sliced[:limit]
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
                sliced = frames[start::step]
                if limit: sliced = sliced[:limit]
                views[cid] = sliced
    return dict(views)


def parse_args():
    parser = argparse.ArgumentParser(description="MASt3R-to-GGPT Data Preparation Script")
    add_dataset_args(parser)
    parser.add_argument("--views", nargs="+", type=int, help="Optional view counts to run (e.g. 2 3 4)")
    parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/mast3r/ggpt_inputs"),
                        help="Root output directory")
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun logging")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_type = args.data
    dataset_cfg = DATASETS[dataset_type]

    # Resolve subjects to process
    subjects_to_process, subjects_codes = get_selected_subjects(args)
    print(f"[INFO] Processing subjects: {subjects_codes}")

    # 2. Load Model
    print(f"[INFO] Loading model {MODEL_NAME} on {DEVICE}...")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    for subject_name, subj_code in zip(subjects_to_process, subjects_codes):
        dataset_root = os.path.join(dataset_cfg["root"], subject_name)
        if not os.path.isdir(dataset_root):
            print(f"[ERROR] Subject directory not found: {dataset_root}")
            continue

        print(f"\n" + "=" * 50)
        print(f"PROCESSING SUBJECT: {subj_code} ({subject_name})")
        print("=" * 50)

        # 4. Initialize Rerun
        if not args.no_rerun and rr is not None:
            rr.init(f"ggpt_prep_{dataset_type}_{subj_code}", spawn=False)
            rr.connect_grpc(RERUN_ADDR)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        # Determine view counts to run
        if args.views:
            camera_counts = args.views
        else:
            view_configs = dataset_cfg.get("view_configs", {})
            if dataset_type == "hi4d":
                pair_prefix = subject_name.split("/")[0]
                cfg = view_configs.get(pair_prefix, view_configs.get("default", {}))
                camera_counts = sorted([int(k) for k in cfg.keys()])
            else:
                camera_counts = [2, 3, 4]

        for n_views in camera_counts:
            print(f"\n[INFO] Processing with {n_views} views...")

            # Resolve target views
            view_configs = dataset_cfg.get("view_configs", {})
            if dataset_type == "hi4d":
                pair_prefix = subject_name.split("/")[0]
                cfg = view_configs.get(pair_prefix, view_configs.get("default", {}))
                target_view_names = cfg.get(n_views)
            else:
                target_view_names = view_configs.get(n_views)

            if target_view_names is None:
                target_view_names = dataset_cfg.get("default_target_views")

            views_dict = build_views(dataset_root, target_view_names, dataset_type=dataset_type,
                                     start=args.start, step=args.step, limit=args.limit)
            view_names = sorted(views_dict.keys())
            if not view_names:
                print(f"[ERROR] No views found for target count {n_views}")
                continue

            n_frames = len(views_dict[view_names[0]])
            print(f"[INFO] Using views: {view_names}")

            # Accumulators for this specific run
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

            # Create a temporary directory for global alignment cache
            with tempfile.TemporaryDirectory() as tmp_dir:
                for t in tqdm(range(n_frames), desc=f"Frames ({subj_code} - {n_views}v)"):
                    current_files = [views_dict[v][t] for v in view_names]

                    # Resolve actual frame index (e.g. for Hi4D it might be 000000)
                    first_view_path = current_files[0]
                    frame_filename = os.path.basename(first_view_path)
                    try:
                        actual_t = int(os.path.splitext(frame_filename)[0])
                    except ValueError:
                        actual_t = t

                    # 1. Image Masking (HI4D only)
                    masked_current_files = []
                    for i, v in enumerate(view_names):
                        m_img = get_masked_image(actual_t, v, current_files[i], tmp_dir, dataset_root,
                                                 mask_subjects=args.mask_subjects, dataset_type=dataset_type)
                        masked_current_files.append(m_img)

                    # 2. Inference
                    imgs = load_images(masked_current_files, size=IMAGE_SIZE)
                    pairs = make_pairs(imgs, scene_graph=SCENE_GRAPH, symmetrize=True)

                    frame_cache = os.path.join(tmp_dir, f"t{t:03d}")
                    os.makedirs(frame_cache, exist_ok=True)
                    scene = sparse_global_alignment(
                        masked_current_files,
                        pairs,
                        frame_cache,
                        model,
                        device=DEVICE,
                        matching_conf_thr=0.0
                    )
                    # 3. Extract Data
                    pts3d_list, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH))

                    try:
                        im_poses = scene.get_im_poses()
                    except AttributeError:
                        im_poses = scene.get_poses()
                    c2ws = to_numpy(im_poses)

                    try:
                        est_intrinsics_all = scene.get_intrinsics()
                    except AttributeError:
                        est_intrinsics_all = scene.intrinsics
                    Ks = to_numpy(est_intrinsics_all)

                    depth_max = dataset_cfg.get("depth_max_m", DEPTH_MAX_M)
                    gt_validity_masks = build_gt_validity_masks(actual_t, view_names, dataset_root,
                                                                depth_max_m=depth_max if depth_max is not None else 999.0,
                                                                dataset_type=dataset_type)

                    for i in range(len(view_names)):
                        conf_i = confs[i]
                        pts_i = pts3d_list[i]

                        H_mod, W_mod = conf_i.shape[:2]
                        pts_i = pts_i.reshape(H_mod, W_mod, 3)
                        conf_i = conf_i.reshape(H_mod, W_mod)

                        thr = np.percentile(conf_i, 100 * (1 - CONF_PERCENTILE))
                        conf_mask = conf_i > thr

                        gt_mask = gt_validity_masks[i]
                        if gt_mask is None:
                            gt_mask = np.zeros((H_mod, W_mod), dtype=bool)
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

                        # Images: dust3r normalized [-1, 1], convert to [0, 1]
                        img_i = imgs[i]['img'].squeeze(0).permute(1, 2, 0).cpu().numpy()
                        img_i = (img_i + 1.0) / 2.0
                        all_images.append(img_i)

                        # Extrinsics: GGPT wants w2c
                        w2c_i = np.linalg.inv(c2ws[i])
                        all_extrinsics.append(w2c_i)
                        all_intrinsics.append(Ks[i])
                        all_point_masks.append(final_mask)

                        # GT Data
                        if dataset_type == "hi4d":
                            view_dir = os.path.join(dataset_root, "images", view_names[i])
                        else:
                            view_dir = os.path.join(dataset_root, view_names[i])

                        K_gt, cam2world_gt = load_gt_params(view_dir, dataset_type=dataset_type)
                        w2c_gt = np.linalg.inv(cam2world_gt)
                        all_gt_extrinsics.append(w2c_gt)

                        # Depth processing for GT points
                        if dataset_type == "dex-ycb":
                            depth_path = os.path.join(view_dir, "depth", f"{actual_t:05d}.png")
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
                                pts_world_gt = (cam2world_gt[:3, :3] @ pts_cam_gt.reshape(-1, 3).T).T + cam2world_gt[
                                    :3, 3]
                                pts_world_gt = pts_world_gt.reshape(H_mod, W_mod, 3)
                                gt_mask_small = (depth_small > 0) & (depth_small <= depth_max)
                                pts_world_gt[~gt_mask_small] = 0
                                all_gt_points.append(pts_world_gt)
                                all_gt_masks.append(gt_mask_small)
                            else:
                                all_gt_points.append(np.zeros((H_mod, W_mod, 3), dtype=np.float32))
                                all_gt_masks.append(np.zeros((H_mod, W_mod), dtype=bool))
                                all_gt_intrinsics.append(K_gt)
                        elif dataset_type == "hi4d":
                            # For HI4D, we don't have per-view depth maps, but we can project the GT mesh
                            # However, for GGPT inputs, we might just want to store the mesh if needed,
                            # or use the build_gt_pointcloud later.
                            # For now, let's just store Identity or zero as placeholder for gt.bin if not strictly needed
                            all_gt_points.append(np.zeros((H_mod, W_mod, 3), dtype=np.float32))
                            all_gt_masks.append(np.zeros((H_mod, W_mod), dtype=bool))
                            # But we DO need intrinsics for evaluation
                            rgb_path = current_files[i]
                            img_orig = cv2.imread(rgb_path)
                            if img_orig is not None:
                                H_orig, W_orig = img_orig.shape[:2]
                                K_small = K_gt.copy()
                                K_small[0, 0] *= (W_mod / W_orig)
                                K_small[1, 1] *= (H_mod / H_orig)
                                K_small[0, 2] *= (W_mod / W_orig)
                                K_small[1, 2] *= (H_mod / H_orig)
                                all_gt_intrinsics.append(K_small)
                            else:
                                all_gt_intrinsics.append(K_gt)

                    # ── Rerun Logging ────────────────────────────────────────────────
                    if not args.no_rerun and rr is not None:
                        rr.set_time("frame", sequence=t)
                        f_m_pts = []
                        for i in range(-len(view_names), 0):
                            m_pts = all_points[i]
                            m_mask = all_point_masks[i]
                            if m_mask.any(): f_m_pts.append(m_pts[m_mask])
                        if f_m_pts:
                            rr.log(f"world/subject_{subj_code}/points",
                                   rr.Points3D(np.concatenate(f_m_pts), colors=[255, 200, 0], radii=0.002))

                        gt_cloud = build_gt_pointcloud(actual_t, view_names, dataset_root, dataset_type=dataset_type)
                        if gt_cloud is not None:
                            rr.log(f"world/subject_{subj_code}/gt_points",
                                   rr.Points3D(gt_cloud, colors=[0, 255, 100], radii=0.002))

            # ── Save Bins for this view count ──────────────────────────────────
            dataset_out_name = dataset_type if dataset_type != "dex-ycb" else "dex-ycb"
            out_path = os.path.join(args.output_dir, dataset_out_name, f"subject-{subj_code}", f"{n_views}views")
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

        print(f"[SUCCESS] Export complete for subject {subj_code}.")


if __name__ == "__main__":
    main()
