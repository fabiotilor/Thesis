import numpy as np
import os
import cv2
import pickle
from pi3.utils.camera_utils import get_rgb_path

DEPTH_MAX_M = 1.5
DEPTH_SCALE = 0.001  # mm → metres


def load_gt_params(view_dir, dataset_type="dex-ycb"):
    if dataset_type == "dex-ycb":
        data = np.load(os.path.join(view_dir, "intrinsics_extrinsics.npz"))
        K = data['intrinsics'].astype(np.float64)[:3, :3]
        cam2world = np.linalg.inv(data['extrinsics'].astype(np.float64))
        return K, cam2world
    elif dataset_type == "hi4d":
        # For Hi4D, view_dir is usually .../pairXX/actionXX/ID
        # The cameras are in .../pairXX/actionXX/cameras/rgb_cameras.npz
        action_dir = os.path.dirname(view_dir)
        cam_id = os.path.basename(view_dir)
        cam_path = os.path.join(action_dir, "cameras", "rgb_cameras.npz")
        if not os.path.exists(cam_path):
            # Try one level higher if needed
            action_dir = os.path.dirname(action_dir)
            cam_path = os.path.join(action_dir, "cameras", "rgb_cameras.npz")

        data = np.load(cam_path)
        ids = list(data['ids'])
        try:
            idx = ids.index(int(cam_id))
        except ValueError:
            # Fallback for string comparison
            idx = ids.index(str(cam_id))

        K = data['intrinsics'][idx].astype(np.float64)
        ext = data['extrinsics'][idx].astype(np.float64)  # 3x4
        # Hi4D extrinsics are world-to-camera [R|t]
        # We need cam2world
        R = ext[:3, :3]
        t = ext[:3, 3]
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        return K, c2w
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def backproject(depth_m, K):
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cx_val = K[0, 2], K[1, 2]  # renamed to cx_val to avoid conflict if needed, wait cx is cx
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = (depth_m > 0) & (depth_m < DEPTH_MAX_M)
    z = depth_m[mask]
    pts = np.stack([(u[mask] - K[0, 2]) * z / fx, (v[mask] - K[1, 2]) * z / fy, z], axis=-1)
    return pts, mask


def _load_hi4d_mesh_gt(dataset_root, t):
    """Load raw mesh scan from frames_vis as GT pointcloud for Hi4D."""
    pkl_path = os.path.join(dataset_root, "frames_vis", f"mesh-f{t:05d}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            mesh = pickle.load(f)
        return mesh['vertices']  # (42930, 3) float32

    # Fallback to SMPL if mesh not available
    smpl_path = os.path.join(dataset_root, "smpl", f"{t:06d}.npz")
    if os.path.exists(smpl_path):
        data = np.load(smpl_path)
        verts = data['verts']  # 2 x 6890 x 3
        return verts.reshape(-1, 3)

    return None


def _load_hi4d_seg_mask(dataset_root, cam_id, t):
    """Load the combined (all-subject) segmentation mask for a Hi4D camera+frame.
    Returns a bool mask where True = person pixel, False = background.
    """
    mask_path = os.path.join(dataset_root, "seg", "img_seg_mask", str(cam_id), "all", f"{t:06d}.png")
    if os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return m > 0  # person pixels = True
    return None


def build_gt_pointcloud(t, view_names, dataset_root, dataset_type="dex-ycb"):
    if dataset_type == "hi4d":
        return _load_hi4d_mesh_gt(dataset_root, t)

    all_pts = []
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw * DEPTH_SCALE

        K, cam2world = load_gt_params(view_dir, dataset_type=dataset_type)
        pts_cam, _ = backproject(depth_m, K)

        pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
        all_pts.append(pts_world)

    return np.concatenate(all_pts, axis=0) if all_pts else None


def get_camera_correspondences(t, view_names, est_poses_all, dataset_root, dataset_type="dex-ycb"):
    est_positions, gt_positions = [], []
    for i, vname in enumerate(view_names):
        c2w_est = est_poses_all[i]
        est_positions.append(c2w_est[:3, 3])
        view_dir = os.path.join(dataset_root, vname)
        _, cam2world_gt = load_gt_params(view_dir, dataset_type=dataset_type)
        gt_positions.append(cam2world_gt[:3, 3])
    return np.array(est_positions), np.array(gt_positions)


def build_static_gt_pointcloud(t, view_names, dataset_root, precomputed_masks=None, dataset_type="dex-ycb"):
    if dataset_type == "hi4d":
        # Hi4D has no static background in GT — return the full mesh GT
        # (used for temporal alignment in strategies 1-3)
        return build_gt_pointcloud(t, view_names, dataset_root, dataset_type=dataset_type)

    all_pts = []
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)

        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw * DEPTH_SCALE
        H, W = depth_m.shape

        rgb_t = get_rgb_path(view_dir, t, dataset_type=dataset_type)
        rgb_adj = get_rgb_path(view_dir, t + 1, dataset_type=dataset_type) or get_rgb_path(view_dir, t - 1,
                                                                                           dataset_type=dataset_type)

        if precomputed_masks is not None and vname in precomputed_masks and precomputed_masks[vname] is not None:
            static_mask = precomputed_masks[vname]
            if static_mask.shape != (H, W):
                static_mask = cv2.resize(static_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(
                    bool)
        else:
            static_mask = np.ones((H, W), dtype=bool)
            if rgb_t is not None and rgb_adj is not None:
                from pi3.utils.optical_flow import compute_flow_sam2
                f0 = cv2.imread(rgb_t)
                f1 = cv2.imread(rgb_adj)
                if f0 is not None and f1 is not None and f0.shape == f1.shape:
                    flow_mask = compute_flow_sam2([f0, f1])
                    if flow_mask.shape != (H, W):
                        static_mask = cv2.resize(
                            flow_mask.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        static_mask = flow_mask

        _dbg_out = os.path.join("flow_masks_output", vname)
        os.makedirs(_dbg_out, exist_ok=True)
        cv2.imwrite(os.path.join(_dbg_out, f"gt_mask_used_{t:05d}.png"),
                    static_mask.astype(np.uint8) * 255)

        K, cam2world = load_gt_params(view_dir, dataset_type=dataset_type)
        keep = (depth_m > 0) & static_mask
        ys, xs = np.where(keep)
        z = depth_m[ys, xs]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=-1)
        pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
        all_pts.append(pts_world)

    return np.concatenate(all_pts, axis=0) if all_pts else None


from eval_config import CONF_PERCENTILE


def get_static_correspondences(t, view_names, pts3d_list, confs, dataset_root,
                               conf_percentile=CONF_PERCENTILE,
                               precomputed_masks=None,
                               use_static_mask=True,
                               dataset_type="dex-ycb"):
    if dataset_type == "hi4d":
        return _get_correspondences_hi4d(
            t, view_names, pts3d_list, confs, dataset_root,
            conf_percentile=conf_percentile,
        )

    all_est = []
    all_gt = []

    # ── Global threshold for the whole frame ──
    all_confs = np.concatenate([c.ravel() for c in confs])
    frame_thr = np.quantile(all_confs, 1.0 - conf_percentile)

    for i, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, vname)

        # ── GT depth (original sensor resolution) ─────────────────────────────
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_gt = depth_raw * DEPTH_SCALE
        H, W = depth_gt.shape

        # ── Model outputs at native model resolution ──────────────────────────
        p3d_model = pts3d_list[i]  # (h_mod*w_mod, 3)  or  (h_mod, w_mod, 3)
        conf_mod = confs[i]  # (h_mod, w_mod)    or  (h_mod*w_mod,)

        # Recover 2-D spatial shape
        if conf_mod.ndim == 1:
            side = int(np.sqrt(len(conf_mod)))
            h_mod, w_mod = side, side
        else:
            h_mod, w_mod = conf_mod.shape

        if p3d_model.ndim == 2:
            p3d_model = p3d_model.reshape(h_mod, w_mod, 3)
        conf_mod = conf_mod.reshape(h_mod, w_mod)

        # ── Scale K to match the model resolution ─────────────────────────────
        K, cam2world = load_gt_params(view_dir, dataset_type=dataset_type)
        scale_x = w_mod / W
        scale_y = h_mod / H
        fx_s = K[0, 0] * scale_x
        fy_s = K[1, 1] * scale_y
        cx_s = K[0, 2] * scale_x
        cy_s = K[1, 2] * scale_y

        # ── Downsample GT depth to model resolution (INTER_NEAREST = no blending) ──
        depth_small = cv2.resize(depth_gt, (w_mod, h_mod),
                                 interpolation=cv2.INTER_NEAREST)

        # ── Farneback static mask, downsampled to model resolution ─────────────
        rgb_t = get_rgb_path(view_dir, t, dataset_type=dataset_type)
        rgb_adj = get_rgb_path(view_dir, t + 1, dataset_type=dataset_type) or get_rgb_path(view_dir, t - 1,
                                                                                           dataset_type=dataset_type)

        if precomputed_masks is not None and vname in precomputed_masks and precomputed_masks[vname] is not None:
            static_small = cv2.resize(precomputed_masks[vname].astype(np.uint8), (w_mod, h_mod),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            static_small = np.ones((h_mod, w_mod), dtype=bool)
            if rgb_t is not None and rgb_adj is not None:
                from pi3.utils.optical_flow import compute_flow_sam2
                f0 = cv2.imread(rgb_t)
                f1 = cv2.imread(rgb_adj)
                if f0 is not None and f1 is not None and f0.shape == f1.shape:
                    flow_mask = compute_flow_sam2([f0, f1])
                    static_small = cv2.resize(
                        flow_mask.astype(np.uint8), (w_mod, h_mod),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

        # ── Valid pixel mask (all criteria at model resolution) ────────────────
        valid = (conf_mod > frame_thr) \
                & (depth_small > 0) \
                & (depth_small < DEPTH_MAX_M)

        if use_static_mask:
            valid &= static_small

        if not np.any(valid):
            continue

        ys, xs = np.where(valid)

        # ── GT pointmap: backproject downsampled depth with scaled K ───────────
        z_gt = depth_small[ys, xs]
        pts_cam_gt = np.stack([(xs - cx_s) * z_gt / fx_s,
                               (ys - cy_s) * z_gt / fy_s,
                               z_gt], axis=-1)
        pts_world_gt = (cam2world[:3, :3] @ pts_cam_gt.T).T + cam2world[:3, 3]

        # ── Estimated pointmap: read directly — no resize, no interpolation ────
        pts_model = p3d_model[ys, xs]

        all_est.append(pts_model)
        all_gt.append(pts_world_gt)

    if not all_est:
        return None, None

    return np.concatenate(all_est, axis=0), np.concatenate(all_gt, axis=0)


def _get_correspondences_hi4d(t, view_names, pts3d_list, confs, dataset_root,
                              conf_percentile=CONF_PERCENTILE):
    """
    Hi4D correspondence extraction without depth maps.

    Project GT mesh into the 2D image plane to establish 2D-3D correspondences
    with the DUSt3R estimated points.
    """
    # ── GT mesh vertices ──
    gt_pts = _load_hi4d_mesh_gt(dataset_root, t)
    if gt_pts is None or len(gt_pts) == 0:
        return None, None

    # ── Global confidence threshold ──
    all_confs = np.concatenate([c.ravel() for c in confs])
    frame_thr = np.quantile(all_confs, 1.0 - conf_percentile)

    src_pts = []
    dst_pts = []

    for i, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, vname)
        pts = pts3d_list[i]
        conf = confs[i]

        # Recover spatial shape
        if conf.ndim == 1:
            side = int(np.sqrt(len(conf)))
            h_mod, w_mod = side, side
        else:
            h_mod, w_mod = conf.shape

        if pts.ndim == 2:
            pts = pts.reshape(h_mod, w_mod, 3)
        conf = conf.reshape(h_mod, w_mod)

        # Load segmentation mask to keep only person pixels
        seg_mask = _load_hi4d_seg_mask(dataset_root, vname, t)
        if seg_mask is not None:
            seg_small = cv2.resize(
                seg_mask.astype(np.uint8), (w_mod, h_mod),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            seg_small = np.ones((h_mod, w_mod), dtype=bool)

        valid = (conf > frame_thr) & seg_small
        if not np.any(valid):
            continue

        # Get camera parameters
        K, cam2world = load_gt_params(view_dir, dataset_type="hi4d")

        # Get original image size to scale K
        rgb_path = get_rgb_path(view_dir, t, dataset_type="hi4d")
        if rgb_path and os.path.exists(rgb_path):
            img = cv2.imread(rgb_path)
            H_orig, W_orig = img.shape[:2]
        else:
            H_orig, W_orig = 940, 1280  # Common Hi4D fallback

        scale_x = w_mod / W_orig
        scale_y = h_mod / H_orig
        K_scaled = np.array([
            [K[0, 0] * scale_x, 0, K[0, 2] * scale_x],
            [0, K[1, 1] * scale_y, K[1, 2] * scale_y],
            [0, 0, 1]
        ])

        # Project GT mesh to camera coordinates
        world2cam = np.linalg.inv(cam2world)
        pts_cam_gt = (world2cam[:3, :3] @ gt_pts.T).T + world2cam[:3, 3]

        # Filter points behind camera
        valid_z_mask = pts_cam_gt[:, 2] > 0
        pts_cam_gt_valid = pts_cam_gt[valid_z_mask]
        gt_pts_valid = gt_pts[valid_z_mask]

        if len(pts_cam_gt_valid) == 0:
            continue

        # Project to 2D
        uvz = (K_scaled @ pts_cam_gt_valid.T).T
        u = np.round(uvz[:, 0] / uvz[:, 2]).astype(int)
        v = np.round(uvz[:, 1] / uvz[:, 2]).astype(int)
        z = pts_cam_gt_valid[:, 2]

        # Create depth buffer to handle occlusions
        depth_buffer = np.full((h_mod, w_mod), np.inf)
        index_buffer = np.full((h_mod, w_mod), -1, dtype=int)

        # Fast vectorised filtering for bounds
        in_bounds = (u >= 0) & (u < w_mod) & (v >= 0) & (v < h_mod)
        u_in = u[in_bounds]
        v_in = v[in_bounds]
        z_in = z[in_bounds]
        idx_in = np.arange(len(u))[in_bounds]

        # Iterative update for occlusion handling (closest z)
        # Note: A pure numpy advanced indexing approach might not correctly
        # keep the *minimum* z if multiple points map to the same pixel.
        for ui, vi, zi, idxi in zip(u_in, v_in, z_in, idx_in):
            if zi < depth_buffer[vi, ui]:
                depth_buffer[vi, ui] = zi
                index_buffer[vi, ui] = idxi

        # Final filtering: find valid pixels that have both DUSt3R estimation and GT projection
        ys, xs = np.where(valid & (index_buffer != -1))

        if len(ys) < 3:
            continue

        src_pts.append(pts[ys, xs])
        dst_pts.append(gt_pts_valid[index_buffer[ys, xs]])

    if not src_pts:
        return None, None

    return np.concatenate(src_pts, axis=0), np.concatenate(dst_pts, axis=0)


def build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=1.5, target_hw=None, cache=None,
                            dataset_type="dex-ycb"):
    # Check cache for this specific frame
    if cache is not None and t in cache:
        return cache[t]

    masks = []

    if dataset_type == "hi4d":
        # Hi4D has no depth maps — use segmentation masks instead.
        # Person pixels = valid (since GT is person mesh).
        for vname in view_names:
            seg_mask = _load_hi4d_seg_mask(dataset_root, vname, t)
            if seg_mask is not None:
                if target_hw is not None and seg_mask.shape != tuple(target_hw):
                    seg_mask = cv2.resize(
                        seg_mask.astype(np.uint8),
                        (target_hw[1], target_hw[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                masks.append(seg_mask)
            else:
                # No seg mask — accept all pixels
                masks.append(None)
        if cache is not None:
            cache[t] = masks
        return masks

    # Dex-YCB: depth-based validity masks
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")

        if not os.path.exists(depth_path):
            masks.append(None)
            continue

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float32) / 1000.0  # mm → m

        mask = (depth_m > 0) & (depth_m <= depth_max_m)  # (H, W) bool

        if target_hw is not None and mask.shape != tuple(target_hw):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_hw[1], target_hw[0]),  # cv2 wants (W, H)
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        masks.append(mask)
    if cache is not None:
        cache[t] = masks
    return masks
