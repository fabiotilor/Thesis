import numpy as np
import os
import cv2
from pi3.utils.camera_utils import get_rgb_path

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


def build_gt_pointcloud(t, view_names, dataset_root):
    all_pts = []
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw * DEPTH_SCALE

        K, cam2world = load_gt_params(view_dir)
        pts_cam, _ = backproject(depth_m, K)

        pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
        all_pts.append(pts_world)

    return np.concatenate(all_pts, axis=0) if all_pts else None


def get_camera_correspondences(t, view_names, est_poses_all, dataset_root):
    est_positions, gt_positions = [], []
    for i, vname in enumerate(view_names):
        c2w_est = est_poses_all[i]
        est_positions.append(c2w_est[:3, 3])
        view_dir = os.path.join(dataset_root, vname)
        _, cam2world_gt = load_gt_params(view_dir)
        gt_positions.append(cam2world_gt[:3, 3])
    return np.array(est_positions), np.array(gt_positions)


def build_static_gt_pointcloud(t, view_names, dataset_root, precomputed_masks=None):
    all_pts = []

    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)

        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw * DEPTH_SCALE
        H, W = depth_m.shape

        rgb_t   = get_rgb_path(view_dir, t)
        rgb_adj = get_rgb_path(view_dir, t + 1) or get_rgb_path(view_dir, t - 1)

        if precomputed_masks is not None and vname in precomputed_masks and precomputed_masks[vname] is not None:
            static_mask = precomputed_masks[vname]
            if static_mask.shape != (H, W):
                static_mask = cv2.resize(static_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            static_mask = np.ones((H, W), dtype=bool)
            if rgb_t is not None and rgb_adj is not None:
                from utils.optical_flow import compute_flow_sam2
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

        K, cam2world = load_gt_params(view_dir)
        keep = (depth_m > 0) & static_mask
        ys, xs = np.where(keep)
        z = depth_m[ys, xs]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_cam   = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=-1)
        pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
        all_pts.append(pts_world)

    return np.concatenate(all_pts, axis=0) if all_pts else None

from eval_config import MIN_CONF_THR
def get_static_correspondences(t, view_names, pts3d_list, confs, dataset_root,
                               min_conf_thr=MIN_CONF_THR,
                               precomputed_masks=None):
    """
    Build (estimated, GT) 3-D point correspondences for static regions.

    Strategy — work at native Pi3 pointmap resolution to avoid interpolating
    3-D points:

      1. Get Pi3 pointmap  p3d[h_mod, w_mod, 3]  and conf[h_mod, w_mod].
         These are already at the model's internal resolution — no resize needed.

      2. Downsample GT depth  (H, W) → (h_mod, w_mod)  with INTER_NEAREST so
         depth values are never blended, and scale K accordingly.

      3. Downsample the Farneback static mask the same way.

      4. Backproject the downsampled GT depth at every valid pixel using the
         scaled K  →  GT pointmap at model resolution.

      5. Read the Pi3 pointmap at the same pixels  →  estimated pointmap.

      6. The paired (estimated, GT) 3-D points are the correspondences.
    """
    all_est = []
    all_gt  = []

    for i, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, vname)

        # ── GT depth (original sensor resolution) ─────────────────────────────
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_gt  = depth_raw * DEPTH_SCALE
        H, W      = depth_gt.shape

        # ── Model outputs at native model resolution ──────────────────────────
        p3d_model = pts3d_list[i]   # (h_mod*w_mod, 3)  or  (h_mod, w_mod, 3)
        conf_mod  = confs[i]        # (h_mod, w_mod)    or  (h_mod*w_mod,)

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
        K, cam2world = load_gt_params(view_dir)
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
        rgb_t   = get_rgb_path(view_dir, t)
        rgb_adj = get_rgb_path(view_dir, t + 1) or get_rgb_path(view_dir, t - 1)

        if precomputed_masks is not None and vname in precomputed_masks and precomputed_masks[vname] is not None:
            static_small = cv2.resize(precomputed_masks[vname].astype(np.uint8), (w_mod, h_mod), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            static_small = np.ones((h_mod, w_mod), dtype=bool)
            if rgb_t is not None and rgb_adj is not None:
                from utils.optical_flow import compute_flow_sam2
                f0 = cv2.imread(rgb_t)
                f1 = cv2.imread(rgb_adj)
                if f0 is not None and f1 is not None and f0.shape == f1.shape:
                    flow_mask = compute_flow_sam2([f0, f1])
                    static_small = cv2.resize(
                        flow_mask.astype(np.uint8), (w_mod, h_mod),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

        # ── Valid pixel mask (all criteria at model resolution) ────────────────
        valid = (conf_mod   > min_conf_thr) \
              & static_small \
              & (depth_small > 0) \
              & (depth_small < DEPTH_MAX_M)

        if not np.any(valid):
            continue

        ys, xs = np.where(valid)

        # ── GT pointmap: backproject downsampled depth with scaled K ───────────
        z_gt = depth_small[ys, xs]
        pts_cam_gt  = np.stack([(xs - cx_s) * z_gt / fx_s,
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

def build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=1.5, target_hw=None, cache=None):
    """
    Returns a list of boolean 2D masks (one per view), True where the GT depth
    is valid (> 0) and within depth_max_m.  Optionally resized to target_hw=(H,W)
    to match the output pointmap resolution.
    """
    # Check cache for this specific frame
    if cache is not None and t in cache:
        return cache[t]

    masks = []
    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")

        if not os.path.exists(depth_path):
            masks.append(None)
            continue

        # DexYCB depths are uint16, stored in millimetres
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m   = depth_raw.astype(np.float32) / 1000.0  # mm → m

        mask = (depth_m > 0) & (depth_m <= depth_max_m)   # (H, W) bool

        if target_hw is not None and mask.shape != tuple(target_hw):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_hw[1], target_hw[0]),          # cv2 wants (W, H)
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        masks.append(mask)
    if cache is not None:
        cache[t] = masks
    return masks