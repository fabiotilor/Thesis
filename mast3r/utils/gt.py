import numpy as np
import os
import cv2
from dust3r.utils.device import to_numpy

DEPTH_MAX_M = 1.5
DEPTH_SCALE = 0.001  # mm → metres

def load_gt_params(view_dir):
    data = np.load(os.path.join(view_dir, "intrinsics_extrinsics.npz"))
    K = data['intrinsics'].astype(np.float64)[:3, :3]
    cam2world = np.linalg.inv(data['extrinsics'].astype(np.float64))
    return K, cam2world


def backproject(depth_m, K):
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = (depth_m > 0) & (depth_m < DEPTH_MAX_M)
    z = depth_m[mask]
    pts = np.stack([(u[mask] - cx) * z / fx, (v[mask] - cy) * z / fy, z], axis=-1)
    return pts, mask


def build_gt_pointcloud(t, view_names, dataset_root, mask_mode="none"):
    all_pts = []
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw * DEPTH_SCALE
        K, cam2world = load_gt_params(view_dir)
        pts_cam, orig_mask = backproject(depth_m, K)
        pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
        if mask_mode != "none":
            mask_path = os.path.join(view_dir, "mask", f"{t:05d}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(view_dir, "mask", f"{t:06d}.png")
            if os.path.exists(mask_path):
                seg = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if seg.shape[:2] != depth_raw.shape[:2]:
                    seg = cv2.resize(seg, (depth_raw.shape[1], depth_raw.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
                valid_seg = (seg > 0)[orig_mask] if mask_mode == "masked" else (seg == 0)[orig_mask]
                pts_world = pts_world[valid_seg]
        all_pts.append(pts_world)
    return np.concatenate(all_pts, axis=0) if all_pts else None

def get_camera_correspondences(t, view_names, scene, dataset_root):
    est_positions, gt_positions = [], []
    im_poses = scene.get_im_poses()
    for i, vname in enumerate(view_names):
        c2w_est = to_numpy(im_poses[i])
        est_positions.append(c2w_est[:3, 3])
        view_dir = os.path.join(dataset_root, vname)
        _, cam2world_gt = load_gt_params(view_dir)
        gt_positions.append(cam2world_gt[:3, 3])
    return np.array(est_positions), np.array(gt_positions)


def build_static_gt_pointcloud(t, view_names, dataset_root,
                               flow_threshold=2.0,
                               mask_mode="none"):
    """
    Build a GT point cloud using only static background regions.

    Static pixels are defined as those that:
      - Belong to the background (mask == 0, i.e. complement of the segmentation mask)
      - Have low optical flow magnitude between frame t and the adjacent frame

    Args:
        t: frame index
        view_names: list of view directory names
        dataset_root: path to the dataset root
        flow_threshold: max optical flow (px) for a pixel to be considered static
        mask_mode: "none" skips segmentation filtering; "masked"/"inverse_masked" apply it

    Returns:
        np.ndarray: (K, 3) static GT points in world coords, or None if none found.
    """
    all_pts = []

    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)

        # ── Depth ──────────────────────────────────────────────────────────────
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw * DEPTH_SCALE
        H, W = depth_m.shape

        # ── RGB paths for optical flow ─────────────────────────────────────────
        rgb_dir = os.path.join(view_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            rgb_dir = view_dir

        def _rgb_path(frame_t):
            for ext in (".png", ".jpg", ".jpeg"):
                p = os.path.join(rgb_dir, f"{frame_t:05d}{ext}")
                if os.path.exists(p):
                    return p
            return None

        rgb_t = _rgb_path(t)

        # ── Static mask via optical flow ───────────────────────────────────────
        flow_mask = np.ones((H, W), dtype=bool)  # conservative default: all static
        if rgb_t is not None:
            # try t+1, fallback to t-1
            rgb_adj = _rgb_path(t + 1) or _rgb_path(t - 1)
            if rgb_adj is not None:
                f0 = cv2.imread(rgb_t, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                f1 = cv2.imread(rgb_adj, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                if f0.shape == f1.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        f0, f1, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                    flow_mag = np.linalg.norm(flow, axis=-1)
                    # Resize flow_mask to H x W (same as depth)
                    flow_mask_full = (flow_mag < flow_threshold)
                    if flow_mask_full.shape != (H, W):
                        flow_mask = cv2.resize(
                            flow_mask_full.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        flow_mask = flow_mask_full

        # ── Segmentation mask (background = static candidates) ─────────────────
        seg_mask = np.ones((H, W), dtype=bool)  # default: keep all
        mask_path = os.path.join(view_dir, "mask", f"{t:05d}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(view_dir, "mask", f"{t:06d}.png")

        if os.path.exists(mask_path):
            seg_t = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if seg_t.shape[:2] != (H, W):
                seg_t = cv2.resize(seg_t, (W, H), interpolation=cv2.INTER_NEAREST)

            # Define background candidate
            is_background = (seg_t == 0)

            # ── Mask Stability: Compute flow on the segmentation mask itself ──
            # This helps remove background pixels near moving object boundaries.
            mask_stable = np.ones((H, W), dtype=bool)
            t_adj = t + 1 if _rgb_path(t + 1) else (t - 1 if _rgb_path(t - 1) else None)
            if t_adj is not None:
                mask_adj_path = os.path.join(view_dir, "mask", f"{t_adj:05d}.png")
                if not os.path.exists(mask_adj_path):
                    mask_adj_path = os.path.join(view_dir, "mask", f"{t_adj:06d}.png")

                if os.path.exists(mask_adj_path):
                    seg_adj = cv2.imread(mask_adj_path, cv2.IMREAD_GRAYSCALE)
                    if seg_adj.shape[:2] != (H, W):
                        seg_adj = cv2.resize(seg_adj, (W, H), interpolation=cv2.INTER_NEAREST)

                    # Compute flow on the masks (treated as grayscale images)
                    flow_seg = cv2.calcOpticalFlowFarneback(
                        seg_t.astype(np.float32), seg_adj.astype(np.float32), None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                    flow_mag_seg = np.linalg.norm(flow_seg, axis=-1)
                    mask_stable = (flow_mag_seg < flow_threshold)

            # Union Logic:
            # - Background (is_background) is always a candidate
            # - Foreground is only a candidate if it is stable (mask flow and RGB flow)
            # User request: "Take all of the complemented pixels (background).
            # Then to the complement add all pixels in original mask (foreground) that have motion of almost 0."
            seg_mask = is_background | (mask_stable & flow_mask)

        # ── Final Static Mask ──────────────────────────────────────────────────
        static_mask = seg_mask & flow_mask

        # ── Backproject and transform to world ─────────────────────────────────
        K, cam2world = load_gt_params(view_dir)

        # Apply depth validity + static mask
        depth_valid = (depth_m > 0) & (depth_m < DEPTH_MAX_M)
        keep = depth_valid & static_mask

        ys, xs = np.where(keep)
        z = depth_m[ys, xs]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=-1)
        pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
        all_pts.append(pts_world)

    if not all_pts:
        return None
    return np.concatenate(all_pts, axis=0)


def get_static_correspondences(t, view_names, scene, dataset_root,
                               flow_threshold=2.0,
                               min_conf_thr=2.0):
    """
    Collect pixel-to-pixel correspondences (model_3d, gt_3d) for static regions.

    Args:
        t: frame index
        view_names: list of view names
        scene: MASt3R scene object (SparseGA)
        dataset_root: path to dataset
        flow_threshold: flow threshold for static pixels
        min_conf_thr: confidence threshold for model points

    Returns:
        tuple (src_pts, dst_pts): (N, 3) arrays of paired points.
    """
    all_est = []
    all_gt = []

    # Get dense model outputs
    pts3d_list, depth_model, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))

    for i, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, vname)

        # ── Load GT Depth ──────────────────────────────────────────────────────
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_gt = depth_raw * DEPTH_SCALE
        H, W = depth_gt.shape

        # ── Compute Union Static Mask (same as build_static_gt_pointcloud) ──────
        mask_path = os.path.join(view_dir, "mask", f"{t:05d}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(view_dir, "mask", f"{t:06d}.png")

        # Basic RGB Flow (re-compute or assume available)
        # Note: In a real optimized system we would share these masks
        # For now, we replicate the logic to ensure identical masks.
        rgb_dir = os.path.join(view_dir, "rgb")
        if not os.path.isdir(rgb_dir): rgb_dir = view_dir
        def _rgb_path(frame_t):
            for ext in (".png", ".jpg", ".jpeg"):
                p = os.path.join(rgb_dir, f"{frame_t:05d}{ext}")
                if os.path.exists(p): return p
            return None

        rgb_t = _rgb_path(t)
        rgb_adj = _rgb_path(t + 1) or _rgb_path(t - 1)
        rgb_flow_mask = np.ones((H, W), dtype=bool)
        if rgb_t and rgb_adj:
            f0 = cv2.imread(rgb_t, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            f1 = cv2.imread(rgb_adj, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            if f0.shape == f1.shape:
                flow = cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                rgb_flow_mask = (np.linalg.norm(flow, axis=-1) < flow_threshold)

        if os.path.exists(mask_path):
            seg_t = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if seg_t.shape[:2] != (H, W):
                seg_t = cv2.resize(seg_t, (W, H), interpolation=cv2.INTER_NEAREST)

            is_background = (seg_t == 0)

            mask_stable = np.ones((H, W), dtype=bool)
            t_adj = t + 1 if _rgb_path(t + 1) else (t - 1 if _rgb_path(t - 1) else None)
            if t_adj:
                m_adj_path = os.path.join(view_dir, "mask", f"{t_adj:05d}.png")
                if not os.path.exists(m_adj_path): m_adj_path = os.path.join(view_dir, "mask", f"{t_adj:06d}.png")
                if os.path.exists(m_adj_path):
                    seg_adj = cv2.imread(m_adj_path, cv2.IMREAD_GRAYSCALE)
                    if seg_adj.shape[:2] != (H, W): seg_adj = cv2.resize(seg_adj, (W, H), interpolation=cv2.INTER_NEAREST)
                    flow_s = cv2.calcOpticalFlowFarneback(seg_t.astype(np.float32), seg_adj.astype(np.float32), None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mask_stable = (np.linalg.norm(flow_s, axis=-1) < flow_threshold)

            # Union Logic:
            # - Always keep Background (is_background)
            # - Add Foreground (seg_t > 0) IF it is stable
            static_mask = is_background | (mask_stable & rgb_flow_mask)
        else:
            static_mask = rgb_flow_mask

        # ── Collect Valid Correspondence Pixels ───────────────────────────────
        conf = cv2.resize(confs[i], (W, H), interpolation=cv2.INTER_NEAREST)
        valid = (conf > min_conf_thr) & static_mask & (depth_gt > 0) & (depth_gt < DEPTH_MAX_M)

        if not np.any(valid):
            continue

        ys, xs = np.where(valid)

        # GT 3D (World)
        K, cam2world = load_gt_params(view_dir)
        z_gt = depth_gt[ys, xs]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_cam_gt = np.stack([(xs - cx) * z_gt / fx, (ys - cy) * z_gt / fy, z_gt], axis=-1)
        pts_world_gt = (cam2world[:3, :3] @ pts_cam_gt.T).T + cam2world[:3, 3]

        # Model 3D (Model Units)
        # Note: MASt3R pts3d/depth are often (N,) flattened.
        # confs[i] usually retains the 2D shape (H, W).
        p3d_model = pts3d_list[i]
        conf_mod = confs[i]
        if conf_mod.ndim == 1:
            # Fallback if confs are also flattened: assume square
            side = int(np.sqrt(len(conf_mod)))
            h_mod, w_mod = side, side
        else:
            h_mod, w_mod = conf_mod.shape

        if p3d_model.ndim == 2: # (N, 3)
            p3d_model = p3d_model.reshape(h_mod, w_mod, 3)

        if p3d_model.shape[:2] != (H, W):
            p3d_model = cv2.resize(p3d_model, (W, H), interpolation=cv2.INTER_LINEAR)

        pts_model = p3d_model[ys, xs]

        # Final check: ensure same count
        if len(pts_model) == len(pts_world_gt):
            all_est.append(pts_model)
            all_gt.append(pts_world_gt)
        else:
            print(f"[WARN] view_{vname} shape mismatch: model={pts_model.shape}, gt={pts_world_gt.shape}")

    if not all_est:
        return None, None

    src = np.concatenate(all_est, axis=0)
    dst = np.concatenate(all_gt, axis=0)
    return src, dst