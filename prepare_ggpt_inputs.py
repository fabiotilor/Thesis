"""
Pi3/Pi3X → GGPT Data Preparation Script.

Runs Pi3 or Pi3X inference on Dex-YCB multi-view sequences and exports
the results in the exact format expected by GGPT's EvalDataset:
    - ff_outputs.bin   (filtered pointmaps, conf, images, extrinsics, intrinsics)
    - sfm_dlt_outputs.bin  (filtered pointmaps + boolean masks)
    - gt.bin           (GT pointmaps, masks, extrinsics, intrinsics)

Usage:
    python prepare_ggpt_inputs.py --model pi3 --subject all --views 2 3 4
    python prepare_ggpt_inputs.py --model pi3x --subject 01 --views 2
"""

import os
import math
import argparse
import numpy as np
import cv2
import glob
import torch
import torchvision.transforms as T
from PIL import Image

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
    DEVICE,
    RERUN_ADDR,
    RERUN_EYE_UP,
)

from pi3.utils.geometry import recover_intrinsic_from_rays_d
from pi3.utils.umeyama_alignment import estimate_similarity_transform
from pi3.utils.gt import (
    load_gt_params,
    build_gt_validity_masks,
    build_gt_pointcloud,
    DEPTH_MAX_M,
    DEPTH_SCALE,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

PIXEL_LIMIT = 255_000  # Pi3's memory-safe pixel budget


def build_views(dataset_root, target_views=None, dataset_type="dex-ycb"):
    """Discover per-view RGB frame paths."""
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    views = {}

    if dataset_type == "hi4d":
        dirs = (
            [os.path.join(dataset_root, "images", str(v)) for v in target_views]
            if target_views
            else sorted(glob.glob(os.path.join(dataset_root, "images", "*")))
        )
    else:
        dirs = (
            [os.path.join(dataset_root, f"view_{v}") for v in target_views]
            if target_views
            else sorted(glob.glob(os.path.join(dataset_root, "view_*")))
        )
    for vd in dirs:
        if not os.path.isdir(vd):
            continue
        vname = os.path.basename(vd)
        rgb_dir = os.path.join(vd, "rgb")
        search = rgb_dir if os.path.isdir(rgb_dir) else vd
        frames = sorted(
            f
            for f in glob.glob(os.path.join(search, "*"))
            if os.path.splitext(f.lower())[1] in img_exts
        )
        if frames:
            views[vname] = frames
    return views


def compute_target_resolution(first_image_path):
    """
    Compute the Pi3 target (W, H) that:
      - respects PIXEL_LIMIT
      - keeps both dims multiples of 14  (patch_size)
    """
    img = Image.open(first_image_path).convert("RGB")
    W_orig, H_orig = img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14  # TARGET_W, TARGET_H


def load_images_for_frame(file_paths, target_w, target_h, device, dataset_type="dex-ycb", dataset_root=None, view_names=None, actual_t=None):
    """Load, resize, and return a (1, V, 3, H, W) tensor in [0, 1]."""
    tensors = []
    for i, f in enumerate(file_paths):
        img = Image.open(f).convert("RGB")
        img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized)

        # Apply segmentation mask for hi4d to filter out background
        if dataset_type == "hi4d" and dataset_root and view_names and actual_t is not None:
            vname = view_names[i]
            from pi3.utils.gt import _load_hi4d_seg_mask
            seg_mask = _load_hi4d_seg_mask(dataset_root, vname, actual_t)
            if seg_mask is not None:
                mask_resized = cv2.resize(
                    seg_mask.astype(np.uint8),
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                img_np[~mask_resized] = 0

        tensors.append(T.ToTensor()(Image.fromarray(img_np)))
    imgs = torch.stack(tensors).to(device)         # (V, 3, H, W)
    return imgs.unsqueeze(0)                        # (1, V, 3, H, W)


def run_forward_pass(model, model_type, imgs_tensor, device):
    """
    Run Pi3 or Pi3X forward and return the output dict.
    Uses AMP for speed on supported GPUs.
    """
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
        if device == "cuda"
        else torch.float32
    )
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype) if device == "cuda" else torch.no_grad():
            if model_type == "pi3":
                res = model(imgs_tensor)
            else:  # pi3x
                res = model(imgs=imgs_tensor, intrinsics=None, poses=None, depths=None)
    return res


def extract_outputs(res, model_type, view_names, confs_list):
    """
    Extract numpy arrays from the raw model output dict.

    Returns
    -------
    pts3d_list : list[np.ndarray]   – per-view (H, W, 3) world points
    confs      : list[np.ndarray]   – per-view (H, W) confidence (sigmoid)
    est_intrinsics : np.ndarray     – (V, 3, 3) estimated intrinsics
    est_poses      : np.ndarray     – (V, 4, 4) estimated c2w poses
    """
    V = len(view_names)

    # ── Points & confidence ──────────────────────────────────────────────
    pts3d_np = res["points"][0].float().cpu().numpy()          # (V, H, W, 3)
    pts3d_list = [pts3d_np[i] for i in range(V)]

    confs_np = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()  # (V, H, W)
    confs = [confs_np[i] for i in range(V)]

    # ── Intrinsics (recovered from local rays) ───────────────────────────
    rays_d = torch.nn.functional.normalize(res["local_points"], dim=-1)
    K_est = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    est_intrinsics = K_est[0].float().cpu().numpy()            # (V, 3, 3)

    # ── Poses (local→world similarity per view) ─────────────────────────
    # Pi3 outputs camera_poses as c2w 4×4 matrices.
    # However, these can contain scale.  The existing pipeline in
    # align_reconstruction_umeyama.py recovers them via Umeyama between
    # local and global points.  We follow the same strategy for consistency.
    all_confs_flat = np.concatenate([c.ravel() for c in confs])
    frame_thr = np.quantile(all_confs_flat, 1.0 - CONF_PERCENTILE)

    est_poses = np.zeros((V, 4, 4))
    for i in range(V):
        local_pts = res["local_points"][0, i].float().cpu().numpy().reshape(-1, 3)
        global_pts = res["points"][0, i].float().cpu().numpy().reshape(-1, 3)
        valid = confs[i].ravel() > frame_thr
        if np.sum(valid) > 10:
            s_est, R_est, t_est = estimate_similarity_transform(local_pts[valid], global_pts[valid])
            T_mat = np.eye(4)
            T_mat[:3, :3] = s_est * R_est
            T_mat[:3, 3] = t_est
            est_poses[i] = T_mat
        else:
            est_poses[i] = np.eye(4)

    return pts3d_list, confs, est_intrinsics, est_poses


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Pi3/Pi3X → GGPT Data Preparation Script")
    parser.add_argument(
        "--model", type=str, choices=["pi3", "pi3x"], default="pi3",
        help="Model variant to use (default: pi3).",
    )
    parser.add_argument(
        "--data", type=str, choices=["dex-ycb", "hi4d"], default="dex-ycb",
        help="Dataset to process (default: dex-ycb).",
    )
    parser.add_argument(
        "--subject", type=str, default="all",
        help="Subject code (e.g. 01) or 'all'.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all subjects in SUBJECT_BY_CODE.",
    )
    parser.add_argument(
        "--views", nargs="+", type=int, default=[2, 3, 4],
        help="Number of views to use (e.g. 2 3 4).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="ggpt_inputs",
        help="Root output directory (model subdirectory is appended automatically).",
    )
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun logging.")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    dataset_name = args.data
    ds_config = DATASETS[dataset_name]
    DATASET_BASE_ROOT = ds_config["root"]
    SUBJECT_NAMES = ds_config["subject_names"]

    # Reconstruct SUBJECT_BY_CODE based on dataset
    if dataset_name == "dex-ycb":
        SUBJECT_BY_CODE = {name.split("subject-")[1][:2] if "subject-" in name else name: name for name in SUBJECT_NAMES}
    elif dataset_name == "hi4d":
        SUBJECT_BY_CODE = {name.split("/")[-1]: name for name in SUBJECT_NAMES}

    # ── Resolve subjects ─────────────────────────────────────────────────
    if args.all or args.subject == "all":
        subjects_to_process = sorted(SUBJECT_BY_CODE.keys())
        print(f"[INFO] Processing all subjects: {subjects_to_process}")
    else:
        subjects_to_process = [args.subject]

    # ── Load model ───────────────────────────────────────────────────────
    model_type = args.model
    print(f"[INFO] Loading model '{model_type}' on {DEVICE}...")

    if model_type == "pi3":
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained("yyfz233/Pi3").to(DEVICE).eval()
    else:
        from pi3.models.pi3x import Pi3X
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        model.disable_multimodal()
        model = model.to(DEVICE)

    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Per-subject loop ─────────────────────────────────────────────────
    for subj_code in subjects_to_process:
        subject_full = SUBJECT_BY_CODE.get(subj_code, subj_code)
        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_full)
        if not os.path.isdir(dataset_root):
            print(f"[ERROR] Subject directory not found: {dataset_root}")
            continue

        print(f"\n{'=' * 60}")
        print(f"PROCESSING SUBJECT: {subj_code} ({subject_full})")
        print("=" * 60)

        # ── Rerun init (per-subject) ─────────────────────────────────────
        if not args.no_rerun and rr is not None:
            rr.init(f"ggpt_prep_{model_type}_{subj_code}", spawn=False)
            rr.connect_grpc(RERUN_ADDR)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        for n_views in args.views:
            print(f"\n[INFO] Processing with {n_views} views...")

            if dataset_name == "hi4d":
                pair_name = subject_full.split("/")[0]
                vc = ds_config["view_configs"].get(pair_name, ds_config["view_configs"].get("default", {}))
                target_view_names = vc.get(n_views)
            else:
                target_view_names = ds_config["view_configs"].get(n_views)

            if target_view_names is None:
                target_view_names = ds_config["default_target_views"]

            views_dict = build_views(dataset_root, target_view_names, dataset_type=dataset_name)
            view_names = sorted(views_dict.keys())
            if not view_names:
                print(f"[ERROR] No views found for target count {n_views}")
                continue

            n_frames = len(views_dict[view_names[0]])
            print(f"[INFO] Using views: {view_names}, {n_frames} frames")

            # Compute target resolution from first frame
            first_file = views_dict[view_names[0]][0]
            TARGET_W, TARGET_H = compute_target_resolution(first_file)
            print(f"[INFO] Target resolution: {TARGET_W}×{TARGET_H} (pixel budget {PIXEL_LIMIT})")

            # ── Accumulators ─────────────────────────────────────────────
            all_points = []
            all_confs = []
            all_images = []
            all_extrinsics = []
            all_intrinsics = []
            all_point_masks = []
            all_gt_points = []
            all_gt_masks = []
            all_gt_extrinsics = []
            all_gt_intrinsics = []

            if dataset_name == "hi4d":
                start_idx = 21
                stride = 1
                max_available = n_frames
                target_count = 24

                t_indices = []
                curr = start_idx
                while len(t_indices) < target_count and curr < max_available:
                    t_indices.append(curr)
                    curr += stride
            else:
                t_indices = list(range(n_frames))

            for t in tqdm(t_indices, desc=f"Frames ({subj_code} - {n_views}v)"):
                current_files = [views_dict[v][t] for v in view_names]

                frame_filename = os.path.basename(current_files[0])
                actual_t = int(os.path.splitext(frame_filename)[0])

                # ── 1. Forward pass ──────────────────────────────────────
                imgs_tensor = load_images_for_frame(
                    current_files, TARGET_W, TARGET_H, DEVICE,
                    dataset_type=dataset_name, dataset_root=dataset_root,
                    view_names=view_names, actual_t=actual_t
                )
                res = run_forward_pass(model, model_type, imgs_tensor, DEVICE)

                pts3d_list, confs, est_intrinsics, est_poses = extract_outputs(
                    res, model_type, view_names, None,
                )

                # ── 2. GT validity masks ─────────────────────────────────
                H_mod, W_mod = confs[0].shape[:2]
                gt_validity_masks = build_gt_validity_masks(
                    actual_t, view_names, dataset_root, depth_max_m=DEPTH_MAX_M, dataset_type=dataset_name,
                )

                # ── 3. Per-view filtering & accumulation ─────────────────
                for i in range(len(view_names)):
                    pts_i = pts3d_list[i].copy()          # (H, W, 3)
                    conf_i = confs[i].copy()               # (H, W)

                    # Confidence percentile threshold
                    thr = np.percentile(conf_i, 100 * (1 - CONF_PERCENTILE))
                    conf_mask = conf_i > thr

                    # GT depth validity mask
                    gt_mask = gt_validity_masks[i]
                    if gt_mask is None:
                        gt_mask = np.zeros((H_mod, W_mod), dtype=bool)
                        print(f"  [WARN] No GT depth for {view_names[i]} at actual_t={actual_t}")
                    else:
                        if gt_mask.shape != (H_mod, W_mod):
                            gt_mask = cv2.resize(
                                gt_mask.astype(np.uint8), (W_mod, H_mod),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)

                    # Combined mask
                    final_mask = conf_mask & gt_mask
                    pts_i[~final_mask] = 0
                    conf_i[~final_mask] = 0

                    all_points.append(pts_i)
                    all_confs.append(conf_i)
                    all_point_masks.append(final_mask)

                    # Image (already [0, 1] float from ToTensor)
                    img_i = imgs_tensor[0, i].permute(1, 2, 0).cpu().float().numpy()
                    all_images.append(img_i)

                    # Extrinsics: GGPT expects W2C (4×4)
                    c2w_i = est_poses[i]
                    w2c_i = np.linalg.inv(c2w_i) if np.linalg.det(c2w_i[:3, :3]) > 1e-6 else np.eye(4)
                    all_extrinsics.append(w2c_i)

                    # Intrinsics
                    all_intrinsics.append(est_intrinsics[i])

                    # ── GT data ──────────────────────────────────────────
                    if dataset_name == "hi4d":
                        view_dir = os.path.join(dataset_root, "images", view_names[i])
                    else:
                        view_dir = os.path.join(dataset_root, view_names[i])
                    K_gt, cam2world_gt = load_gt_params(view_dir, dataset_type=dataset_name)
                    w2c_gt = np.linalg.inv(cam2world_gt)
                    all_gt_extrinsics.append(w2c_gt)

                    depth_path = os.path.join(view_dir, "depth", f"{actual_t:05d}.png")
                    if os.path.exists(depth_path):
                        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                        depth_m = depth_raw * DEPTH_SCALE
                        H_orig, W_orig = depth_raw.shape[:2]

                        # Scale GT intrinsics to model resolution
                        K_small = K_gt.copy()
                        K_small[0, 0] *= (W_mod / W_orig)
                        K_small[1, 1] *= (H_mod / H_orig)
                        K_small[0, 2] *= (W_mod / W_orig)
                        K_small[1, 2] *= (H_mod / H_orig)
                        all_gt_intrinsics.append(K_small)

                        # Backproject GT depth → world points at model resolution
                        depth_small = cv2.resize(depth_m, (W_mod, H_mod), interpolation=cv2.INTER_NEAREST)
                        fy, fx = K_small[1, 1], K_small[0, 0]
                        cy, cx = K_small[1, 2], K_small[0, 2]
                        v_grid, u_grid = np.meshgrid(np.arange(H_mod), np.arange(W_mod), indexing="ij")
                        pts_cam_gt = np.stack(
                            [
                                (u_grid - cx) * depth_small / fx,
                                (v_grid - cy) * depth_small / fy,
                                depth_small,
                            ],
                            axis=-1,
                        )
                        pts_world_gt = (
                            (cam2world_gt[:3, :3] @ pts_cam_gt.reshape(-1, 3).T).T
                            + cam2world_gt[:3, 3]
                        ).reshape(H_mod, W_mod, 3)
                        gt_mask_small = (depth_small > 0) & (depth_small <= DEPTH_MAX_M)
                        pts_world_gt[~gt_mask_small] = 0
                        all_gt_points.append(pts_world_gt)
                        all_gt_masks.append(gt_mask_small)
                    else:
                        if dataset_name != "hi4d":
                            print(f"  [WARN] Missing GT depth: {depth_path}")
                        all_gt_points.append(np.zeros((H_mod, W_mod, 3), dtype=np.float32))
                        all_gt_masks.append(np.zeros((H_mod, W_mod), dtype=bool))
                        K_small = K_gt.copy()
                        with Image.open(current_files[i]) as img_gt:
                            W_orig, H_orig = img_gt.size
                        K_small[0, 0] *= (W_mod / W_orig)
                        K_small[1, 1] *= (H_mod / H_orig)
                        K_small[0, 2] *= (W_mod / W_orig)
                        K_small[1, 2] *= (H_mod / H_orig)
                        all_gt_intrinsics.append(K_small)

                # ── Rerun logging ────────────────────────────────────────
                if not args.no_rerun and rr is not None:
                    rr.set_time("frame", sequence=actual_t)

                    # Model points (filtered)
                    frame_pts = []
                    V = len(view_names)
                    for i in range(-V, 0):
                        m_pts = all_points[i]
                        m_mask = all_point_masks[i]
                        if m_mask.any():
                            frame_pts.append(m_pts[m_mask])
                    if frame_pts:
                        rr.log(
                            f"world/{model_type}/subject_{subj_code}/{n_views}v/est_points",
                            rr.Points3D(np.concatenate(frame_pts), colors=[255, 200, 0], radii=0.002),
                        )

                    # GT points
                    gt_cloud = build_gt_pointcloud(actual_t, view_names, dataset_root, dataset_type=dataset_name)
                    if gt_cloud is not None:
                        rr.log(
                            f"world/{model_type}/subject_{subj_code}/{n_views}v/gt_points",
                            rr.Points3D(gt_cloud, colors=[0, 255, 100], radii=0.002),
                        )

            # ── Save .bin files ───────────────────────────────────────────
            # Output goes to: ggpt_inputs/<dataset_name>/<model>/subject-<code>/<N>views/
            out_path = os.path.join(args.output_dir, dataset_name, model_type, f"subject-{subj_code}", f"{n_views}views")
            os.makedirs(out_path, exist_ok=True)
            print(f"[INFO] Saving results to {out_path}...")

            torch.save(
                {
                    "points": torch.from_numpy(np.stack(all_points)),
                    "points_conf": torch.from_numpy(np.stack(all_confs)),
                    "images_ff": torch.from_numpy(np.stack(all_images)),
                    "extrinsics": torch.from_numpy(np.stack(all_extrinsics)),
                    "intrinsics": torch.from_numpy(np.stack(all_intrinsics)),
                },
                os.path.join(out_path, "ff_outputs.bin"),
            )

            torch.save(
                {
                    "points": torch.from_numpy(np.stack(all_points)),
                    "point_masks": torch.from_numpy(np.stack(all_point_masks)),
                },
                os.path.join(out_path, "sfm_dlt_outputs.bin"),
            )

            torch.save(
                {
                    "points": torch.from_numpy(np.stack(all_gt_points)),
                    "point_masks": torch.from_numpy(np.stack(all_gt_masks)),
                    "extrinsics": torch.from_numpy(np.stack(all_gt_extrinsics)),
                    "intrinsics": torch.from_numpy(np.stack(all_gt_intrinsics)),
                },
                os.path.join(out_path, "gt.bin"),
            )

        print(f"[SUCCESS] Export complete for subject {subj_code}.")

    print("\n[DONE] All subjects processed.")


if __name__ == "__main__":
    main()
