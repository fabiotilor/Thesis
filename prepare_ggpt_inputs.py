import os
import argparse
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
    DATASETS,
    MODEL_NAME,
    DEVICE,
    IMAGE_SIZE,
    RERUN_ADDR,
    RERUN_EYE_UP,
    HI4D_START_FRAME,
    HI4D_STEP_SIZE,
    HI4D_TOTAL_FRAMES,
    get_view_config,
)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.gt import (
    load_gt_params,
    build_gt_validity_masks,
    build_gt_pointcloud,
    DEPTH_MAX_M,
    DEPTH_SCALE,
    _load_hi4d_seg_mask,
)
from vggt.utils.camera_utils import get_rgb_path


def build_views(dataset_root, target_views=None, dataset_type="dex-ycb",
                start=0, step=1, limit=None):
    """Discover per-view frame lists.

    For dex-ycb:  dataset_root/view_XX/rgb/*.png
    For hi4d:     dataset_root/images/<cam_id>/*.jpg
    """
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = {}

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
                if limit:
                    sliced = sliced[:limit]
                views[vname] = sliced

    elif dataset_type == "hi4d":
        img_dir = os.path.join(dataset_root, "images")
        cam_ids = ([str(v) for v in target_views]
                   if target_views else sorted(os.listdir(img_dir)))
        for cid in cam_ids:
            cid_dir = os.path.join(img_dir, cid)
            if not os.path.isdir(cid_dir):
                continue
            frames = sorted(f for f in glob.glob(os.path.join(cid_dir, "*"))
                            if os.path.splitext(f.lower())[1] in img_exts)
            if frames:
                sliced = frames[start::step]
                if limit:
                    sliced = sliced[:limit]
                views[cid] = sliced

    return views


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-to-GGPT Data Preparation Script")
    parser.add_argument("--data", type=str, choices=["dex-ycb", "hi4d"], default="dex-ycb",
                        help="Dataset to use")
    parser.add_argument("--subject", type=str, default="all",
                        help="Subject code (e.g. 01 for dex-ycb) or full name")
    parser.add_argument("--pair", type=str, default=None,
                        help="Specific pair/action for hi4d (e.g. pair00/dance00)")
    parser.add_argument("--all", action="store_true",
                        help="Process all subjects")
    parser.add_argument("--views", nargs="+", type=int, default=None,
                        help="Number of views to use (e.g. 2 3 4)")
    parser.add_argument("--output_dir", type=str, default="ggpt_inputs",
                        help="Root output directory")
    parser.add_argument("--no-rerun", action="store_true",
                        help="Disable Rerun logging")
    # Frame slicing overrides (applied after dataset-specific defaults)
    parser.add_argument("--start", type=int, default=None,
                        help="Start frame index (default: dataset-specific)")
    parser.add_argument("--step", type=int, default=None,
                        help="Frame step size (default: dataset-specific)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of frames (default: dataset-specific)")
    return parser.parse_args()


def _resolve_subjects(args, dataset_cfg, dataset_type):
    """Return list of (subject_full_name, display_code) tuples."""
    subject_names = dataset_cfg["subject_names"]

    if dataset_type == "hi4d":
        # subject_names are like "pair00/dance00"
        subject_by_code = {name: name for name in subject_names}
    else:
        subject_by_code = {name.split("subject-")[1][:2]: name for name in subject_names}

    if args.pair:
        return [(args.pair, args.pair)]

    if args.all or args.subject == "all":
        codes = list(subject_by_code.keys())
    else:
        codes = [args.subject]

    return [(subject_by_code.get(c, c), c) for c in codes]


def _resolve_view_counts(args, dataset_cfg, dataset_type, subject_name):
    """Determine which view counts to iterate over."""
    if args.views:
        return args.views

    view_configs = dataset_cfg.get("view_configs", {})
    if dataset_type == "hi4d":
        pair_prefix = subject_name.split("/")[0]
        cfg = view_configs.get(pair_prefix, view_configs.get("default", {}))
        return sorted([k for k in cfg.keys() if isinstance(k, int)])
    else:
        return [2, 3, 4]


def _resolve_target_views(dataset_cfg, dataset_type, n_views, subject_name):
    """Get the ordered list of camera IDs for a given view count."""
    return get_view_config(dataset_type, n_views,
                           pair_name=subject_name.split("/")[0] if "/" in subject_name else None)


def main():
    args = parse_args()
    dataset_type = args.data
    dataset_cfg = DATASETS[dataset_type]
    dataset_root_base = dataset_cfg["root"]
    depth_max = dataset_cfg.get("depth_max_m")

    # Frame slicing defaults per dataset
    if dataset_type == "hi4d":
        frame_start = args.start if args.start is not None else HI4D_START_FRAME
        frame_step = args.step if args.step is not None else HI4D_STEP_SIZE
        frame_limit = args.limit if args.limit is not None else HI4D_TOTAL_FRAMES
    else:
        frame_start = args.start if args.start is not None else 0
        frame_step = args.step if args.step is not None else 1
        frame_limit = args.limit

    subjects = _resolve_subjects(args, dataset_cfg, dataset_type)
    print(f"[INFO] Dataset: {dataset_type}")
    print(f"[INFO] Subjects: {[c for _, c in subjects]}")
    print(f"[INFO] Frame slicing: start={frame_start}, step={frame_step}, limit={frame_limit}")

    # Load Model
    print(f"[INFO] Loading model {MODEL_NAME} on {DEVICE}...")
    model = VGGT.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    for subject_full, subj_code in subjects:
        dataset_root = os.path.join(dataset_root_base, subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[ERROR] Subject directory not found: {dataset_root}")
            continue

        safe_code = subj_code.replace("/", "_")
        print(f"\n{'=' * 60}")
        print(f"PROCESSING SUBJECT: {subj_code} ({subject_full})")
        print(f"{'=' * 60}")

        # Initialize Rerun
        if not args.no_rerun and rr is not None:
            rr.init(f"ggpt_prep_{dataset_type}_{safe_code}", spawn=False)
            rr.connect_grpc(RERUN_ADDR)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        view_counts = _resolve_view_counts(args, dataset_cfg, dataset_type, subject_full)

        for n_views in view_counts:
            print(f"\n[INFO] Processing with {n_views} views...")

            target_view_names = _resolve_target_views(dataset_cfg, dataset_type, n_views, subject_full)
            if target_view_names is None:
                target_view_names = dataset_cfg.get("default_target_views")

            views_dict = build_views(dataset_root, target_view_names,
                                     dataset_type=dataset_type,
                                     start=frame_start, step=frame_step,
                                     limit=frame_limit)
            view_names = sorted(views_dict.keys())
            if not view_names:
                print(f"[ERROR] No views found for target count {n_views}")
                continue

            n_frames = len(views_dict[view_names[0]])
            print(f"[INFO] Using views: {view_names} ({n_frames} frames)")

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

            # Determine dtype for mixed-precision inference
            dtype = torch.float32
            if DEVICE == "cuda":
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

            for t in tqdm(range(n_frames), desc=f"Frames ({subj_code} - {n_views}v)"):
                current_files = [views_dict[v][t] for v in view_names]

                # Resolve the actual frame index from the filename
                # (e.g. Hi4D filenames are 000022.jpg → actual_t = 22)
                first_frame_path = current_files[0]
                frame_filename = os.path.basename(first_frame_path)
                try:
                    actual_t = int(os.path.splitext(frame_filename)[0])
                except ValueError:
                    actual_t = t

                # Prepare masked files if Hi4D
                import tempfile
                with tempfile.TemporaryDirectory() as run_cache_root:
                    input_files = current_files
                    if dataset_type == "hi4d":
                        masked_files = []
                        for i_v, (v, fpath) in enumerate(zip(view_names, current_files)):
                            seg_mask = _load_hi4d_seg_mask(dataset_root, v, actual_t)
                            if seg_mask is not None:
                                img = cv2.imread(fpath)
                                if img is not None:
                                    H_img, W_img = img.shape[:2]
                                    if seg_mask.shape != (H_img, W_img):
                                        seg_mask_resized = cv2.resize(
                                            seg_mask.astype(np.uint8), (W_img, H_img),
                                            interpolation=cv2.INTER_NEAREST
                                        ).astype(bool)
                                    else:
                                        seg_mask_resized = seg_mask
                                    img[~seg_mask_resized] = 0
                                    tmp_path = os.path.join(run_cache_root, f"masked_{v}_{actual_t:06d}.jpg")
                                    cv2.imwrite(tmp_path, img)
                                    masked_files.append(tmp_path)
                                else:
                                    masked_files.append(fpath)
                            else:
                                masked_files.append(fpath)
                        input_files = masked_files

                    imgs = load_and_preprocess_images(input_files).to(DEVICE)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=dtype) if DEVICE == "cuda" else torch.no_grad():
                            predictions = model(imgs)

                pts3d_all = predictions["world_points"].cpu().float().numpy().squeeze(0)
                confs_all = predictions["world_points_conf"].cpu().float().numpy().squeeze(0)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], imgs.shape[-2:])
                extrinsics_w2c = extrinsic.cpu().float().numpy().squeeze(0)
                intrinsics_est = intrinsic.cpu().float().numpy().squeeze(0)

                # Build validity masks (depth-based for dex-ycb, seg-based for hi4d)
                gt_validity_masks = build_gt_validity_masks(
                    actual_t, view_names, dataset_root,
                    depth_max_m=depth_max if depth_max is not None else 999.0,
                    dataset_type=dataset_type,
                )
                V_count, H_mod, W_mod = confs_all.shape

                for i in range(V_count):
                    pts_i = pts3d_all[i]
                    conf_i = confs_all[i]

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
                    pts_i[~final_mask] = 0
                    conf_i[~final_mask] = 0

                    all_points.append(pts_i)
                    all_confs.append(conf_i)
                    img_i = imgs[i].permute(1, 2, 0).cpu().float().numpy()
                    all_images.append(img_i)

                    ext_4x4 = np.eye(4)
                    ext_4x4[:3, :4] = extrinsics_w2c[i]
                    all_extrinsics.append(ext_4x4)
                    all_intrinsics.append(intrinsics_est[i])
                    all_point_masks.append(final_mask)

                    # ── GT camera parameters ─────────────────────────────────
                    if dataset_type == "hi4d":
                        # For Hi4D, view_dir is dataset_root/<cam_id>
                        # load_gt_params expects the cam_id as basename
                        view_dir = os.path.join(dataset_root, view_names[i])
                    else:
                        view_dir = os.path.join(dataset_root, view_names[i])

                    K_gt, cam2world_gt = load_gt_params(view_dir, dataset_type=dataset_type)
                    w2c_gt = np.linalg.inv(cam2world_gt)
                    all_gt_extrinsics.append(w2c_gt)

                    # ── GT points / intrinsics ───────────────────────────────
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
                                [(u_grid - cx) * depth_small / fx, (v_grid - cy) * depth_small / fy, depth_small], axis=-1)
                            pts_world_gt = (cam2world_gt[:3, :3] @ pts_cam_gt.reshape(-1, 3).T).T + cam2world_gt[:3, 3]
                            pts_world_gt = pts_world_gt.reshape(H_mod, W_mod, 3)
                            gt_mask_small = (depth_small > 0) & (depth_small <= DEPTH_MAX_M)
                            pts_world_gt[~gt_mask_small] = 0
                            all_gt_points.append(pts_world_gt)
                            all_gt_masks.append(gt_mask_small)
                        else:
                            all_gt_points.append(np.zeros((H_mod, W_mod, 3), dtype=np.float32))
                            all_gt_masks.append(np.zeros((H_mod, W_mod), dtype=bool))
                            K_small = K_gt.copy()
                            K_small[0, 0] *= (W_mod / 640)
                            K_small[1, 1] *= (H_mod / 480)
                            K_small[0, 2] *= (W_mod / 640)
                            K_small[1, 2] *= (H_mod / 480)
                            all_gt_intrinsics.append(K_small)

                    elif dataset_type == "hi4d":
                        # Hi4D has no per-view depth maps.
                        # Store zero placeholders for per-pixel GT; the mesh GT
                        # is used downstream via build_gt_pointcloud.
                        all_gt_points.append(np.zeros((H_mod, W_mod, 3), dtype=np.float32))
                        all_gt_masks.append(np.zeros((H_mod, W_mod), dtype=bool))

                        # Scale GT intrinsics to model resolution
                        rgb_path = current_files[i]
                        img_orig = cv2.imread(rgb_path)
                        if img_orig is not None:
                            H_orig, W_orig = img_orig.shape[:2]
                        else:
                            H_orig, W_orig = 940, 1280  # Common Hi4D fallback
                        K_small = K_gt.copy()
                        K_small[0, 0] *= (W_mod / W_orig)
                        K_small[1, 1] *= (H_mod / H_orig)
                        K_small[0, 2] *= (W_mod / W_orig)
                        K_small[1, 2] *= (H_mod / H_orig)
                        all_gt_intrinsics.append(K_small)

                # ── Rerun Logging ────────────────────────────────────────────────
                if not args.no_rerun and rr is not None:
                    rr.set_time("frame", sequence=t)
                    # Model points: aggregate current frame views
                    f_m_pts = []
                    for i in range(-V_count, 0):
                        m_pts = all_points[i]
                        m_mask = all_point_masks[i]
                        if m_mask.any(): f_m_pts.append(m_pts[m_mask])
                    if f_m_pts:
                        rr.log(f"world/subject_{safe_code}/points",
                               rr.Points3D(np.concatenate(f_m_pts), colors=[255, 200, 0], radii=0.002))

                    # GT points
                    gt_cloud = build_gt_pointcloud(actual_t, view_names, dataset_root,
                                                   dataset_type=dataset_type)
                    if gt_cloud is not None:
                        rr.log(f"world/subject_{safe_code}/gt_points",
                               rr.Points3D(gt_cloud, colors=[0, 255, 100], radii=0.002))

            # ── Save Bins for this view count ──────────────────────────────────
            out_path = os.path.join(args.output_dir, dataset_type,
                                    f"subject-{safe_code}", f"{n_views}views")
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