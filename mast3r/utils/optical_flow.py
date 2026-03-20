import os
from collections import defaultdict

import numpy as np
import cv2
import rerun as rr
from dust3r.utils.device import to_numpy


# ── Step 1: Compute static masks ──────────────────────────────────────────────

def compute_static_mask(rgb_paths, flow_threshold=1.0):
    """
    Returns a boolean mask (H, W) where True = static across all frame transitions.
    """
    H, W = cv2.imread(rgb_paths[0], cv2.IMREAD_GRAYSCALE).shape
    static = np.ones((H, W), dtype=bool)

    for i in range(len(rgb_paths) - 1):
        f0 = cv2.imread(rgb_paths[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        f1 = cv2.imread(rgb_paths[i + 1], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # Use Farneback for efficiency and strictness
        flow = cv2.calcOpticalFlowFarneback(
            f0, f1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        static &= (magnitude < flow_threshold)
    return static


# ── Step 2-5: Stabilise Static Points ─────────────────────────────────────────

def stabilise_static_points(
        views,
        view_names,
        all_scenes,
        get_cam_corr_fn,
        estimate_transform_fn,
        log_rerun_fn,
        dataset_root,
        out_dir_in="aligned_outputs",
        out_dir_out="aligned_outputs_stabilised",
        flow_threshold=1.0,
        min_conf_thr=1.5,
        subsample=4,
        verbose=True,
):
    print(f"\n[stabilise] Computing static masks (flow_threshold={flow_threshold}px)...")
    static_masks = {}
    for vname in view_names:
        mask = compute_static_mask(views[vname], flow_threshold)
        static_masks[vname] = mask
        if verbose:
            pct = 100 * mask.mean()
            print(f"  cam {vname}: {pct:.1f}% pixels static")

    T = len(all_scenes)
    os.makedirs(out_dir_out, exist_ok=True)

    # We need to collect positions for each static point across all frames.
    # To do this efficiently, we process each camera separately.
    print(f"[stabilise] Applying temporal median to static points...")

    for cam_idx, vname in enumerate(view_names):
        static_mask = static_masks[vname]

        # We need a shared mapping to match points across frames by pixel index.
        # But wait, confidence masks can change! We only stabilise points that are
        # VALID (conf > thr) across frames. Actually, let's keep it simple:
        # A static point is stabilised IF it is valid in a frame. Its target is
        # the median of its positions in all frames where it WAS valid.

        # 1. Pre-collect valid positions for every pixel in the static mask
        # map: flattened_pixel_idx -> list of 3D positions
        pixel_positions = defaultdict(list)

        for t in range(T):
            # Load the scene for this frame to get the dense map correctly masked
            scene = all_scenes[t]
            pts3d_all, _, confs_all = to_numpy(scene.get_dense_pts3d(clean_depth=True))

            pts3d = pts3d_all[cam_idx]  # (H*W, 3)
            conf = confs_all[cam_idx]  # (H, W)
            H, W = conf.shape
            pts3d = pts3d.reshape(H, W, 3)

            # Resize static mask to match reconstruction resolution (H, W)
            mask_resized = cv2.resize(static_mask.astype(np.uint8), (W, H),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)

            # Load aligned transform for this frame to work in GT world space
            est_cam, gt_cam = get_cam_corr_fn(t, view_names, scene, dataset_root)
            s, R, tr = estimate_transform_fn(est_cam, gt_cam)

            # Identify valid static pixels
            valid_mask = (conf > min_conf_thr) & mask_resized
            valid_idxs = np.where(valid_mask)  # (rows, cols)

            # Transform valid static points to GT world space
            p_mast3r = pts3d[valid_mask]
            p_world = (s * (R @ p_mast3r.T).T) + tr

            # Collect
            flat_idxs = valid_idxs[0] * pts3d.shape[1] + valid_idxs[1]
            for i, f_idx in enumerate(flat_idxs):
                pixel_positions[f_idx].append(p_world[i])

        # 2. Compute medians
        pixel_medians = {}
        for f_idx, positions in pixel_positions.items():
            pixel_medians[f_idx] = np.median(np.array(positions), axis=0)

        # 3. Apply medians to each frame's saved cloud
        for t in range(T):
            # Load the frame
            in_path = os.path.join(out_dir_in, f"frame_{t:02d}.npz")
            data = np.load(in_path)
            aligned_pts = data['aligned_pts'].copy()
            gt_pts = data['gt_pts']
            scale = data['scale']

            # Reconstruct the index mapping for this specific frame
            # MASt3R saved points as: fused_pts = concat(pts[conf > thr][::4])
            # So we need the offset for THIS camera in the fused cloud.
            offset = 0
            scene = all_scenes[t]
            confs_all = to_numpy(scene.get_dense_pts3d(clean_depth=True))[2]
            for prev_cam in range(cam_idx):
                count = (confs_all[prev_cam] > min_conf_thr).sum()
                offset += (count + subsample - 1) // subsample  # result of [::subsample]

            # Now map pixels of THIS camera to indices in ALIGNED_PTS
            conf_cam = confs_all[cam_idx]
            valid_mask_cam = conf_cam > min_conf_thr
            valid_idxs_cam = np.where(valid_mask_cam)
            flat_idxs_cam = valid_idxs_cam[0] * conf_cam.shape[1] + valid_idxs_cam[1]

            # Subsample the same way as building the cloud
            # points are: [0, subsample, 2*subsample, ...]
            # which corresponds to indices in valid_idxs_cam
            sub_idxs = np.arange(0, len(flat_idxs_cam), subsample)

            n_stabilised = 0
            for i_sub in sub_idxs:
                f_idx = flat_idxs_cam[i_sub]
                if f_idx in pixel_medians:
                    p_idx = offset + (i_sub // subsample)
                    aligned_pts[p_idx] = pixel_medians[f_idx]
                    n_stabilised += 1

            # Update and save
            out_path = os.path.join(out_dir_out, f"frame_{t:02d}.npz")
            # If it's the first camera, we overwrite/create. If later, we load and update.
            if cam_idx > 0:
                old_data = np.load(out_path)
                final_aligned = old_data['aligned_pts'].copy()
                # find indices for THIS camera in the old_data
                final_aligned[offset: offset + len(sub_idxs)] = aligned_pts[offset: offset + len(sub_idxs)]
                aligned_pts = final_aligned

            np.savez(out_path, gt_pts=gt_pts, aligned_pts=aligned_pts,
                     scale=scale, frame_idx=int(t))

            if cam_idx == len(view_names) - 1:
                # final pass log
                log_rerun_fn(t, None, None, None, refined_pts=aligned_pts)
                print(f"  t={t:02d}: stabilised points logged to Rerun")

    print(f"[stabilise] Saved stabilised outputs to {out_dir_out}/")