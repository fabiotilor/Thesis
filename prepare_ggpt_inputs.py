import os
import argparse
import torch
import numpy as np
import cv2
import glob
import tempfile
from tqdm import tqdm

try:
    import rerun as rr
except ImportError:
    rr = None

from eval_config import (
    CONF_PERCENTILE,
    DATASET_BASE_ROOT,
    SUBJECT_BY_CODE,
    MODEL_NAME,
    DEVICE,
    IMAGE_SIZE,
    RERUN_ADDR,
    RERUN_EYE_UP,
    VIEW_CONFIGS,
    DEFAULT_TARGET_VIEWS,
    SCENE_GRAPH
)

# MASt3R / DUSt3R imports
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

from mast3r.utils.gt import load_gt_params, build_gt_validity_masks, build_gt_pointcloud, DEPTH_MAX_M, DEPTH_SCALE

CLEAN_DEPTH = True


def build_views(dataset_root, target_views=None):
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = {}
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
    return views


def parse_args():
    parser = argparse.ArgumentParser(description="MASt3R-to-GGPT Data Preparation Script")
    parser.add_argument("--subject", type=str, default="all", help="Subject code (e.g. 01) or full name")
    parser.add_argument("--all", action="store_true", help="Process all subjects in SUBJECT_BY_CODE")
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 4], help="Number of views to use (e.g. 2 3 4)")
    parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/mast3r/ggpt_inputs"),
                        help="Root output directory")
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve subjects to process
    if args.all or args.subject == "all":
        subjects_to_process = sorted(SUBJECT_BY_CODE.keys())
        print(f"[INFO] Processing all subjects: {subjects_to_process}")
    else:
        subjects_to_process = [args.subject]

    # 2. Load Model
    print(f"[INFO] Loading model {MODEL_NAME} on {DEVICE}...")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    for subj_code in subjects_to_process:
        # 1. Resolve subject
        subject_full = SUBJECT_BY_CODE.get(subj_code, subj_code)
        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[ERROR] Subject directory not found: {dataset_root}")
            continue

        print(f"\n" + "=" * 50)
        print(f"PROCESSING SUBJECT: {subj_code} ({subject_full})")
        print("=" * 50)

        # 4. Initialize Rerun
        if not args.no_rerun and rr is not None:
            rr.init(f"ggpt_prep_mast3r_{subj_code}", spawn=False)
            rr.connect_grpc(RERUN_ADDR)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        for n_views in args.views:
            print(f"\n[INFO] Processing with {n_views} views...")

            # Resolve view names from eval_config
            target_view_names = VIEW_CONFIGS.get(n_views)
            if target_view_names is None:
                target_view_names = DEFAULT_TARGET_VIEWS

            views_dict = build_views(dataset_root, target_view_names)
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

                    # 1. Inference
                    imgs = load_images(current_files, size=IMAGE_SIZE)
                    pairs = make_pairs(imgs, scene_graph=SCENE_GRAPH, symmetrize=True)

                    frame_cache = os.path.join(tmp_dir, f"t{t:03d}")
                    os.makedirs(frame_cache, exist_ok=True)
                    scene = sparse_global_alignment(
                        current_files,
                        pairs,
                        frame_cache,
                        model,
                        device=DEVICE,
                        matching_conf_thr=0.0
                    )
                    # 2. Extract Data
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

                    gt_validity_masks = build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=DEPTH_MAX_M)

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
                        view_dir = os.path.join(dataset_root, view_names[i])
                        K_gt, cam2world_gt = load_gt_params(view_dir)
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
                            K_small = K_gt.copy()
                            K_small[0, 0] *= (W_mod / 640)
                            K_small[1, 1] *= (H_mod / 480)
                            K_small[0, 2] *= (W_mod / 640)
                            K_small[1, 2] *= (H_mod / 480)
                            all_gt_intrinsics.append(K_small)

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

                        gt_cloud = build_gt_pointcloud(t, view_names, dataset_root)
                        if gt_cloud is not None:
                            rr.log(f"world/subject_{subj_code}/gt_points",
                                   rr.Points3D(gt_cloud, colors=[0, 255, 100], radii=0.002))

            # ── Save Bins for this view count ──────────────────────────────────
            out_path = os.path.join(args.output_dir, f"subject-{subj_code}", f"{n_views}views")
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
