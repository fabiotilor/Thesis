import numpy as np
import os
import cv2
import torch

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


def get_camera_correspondences(t, view_names, est_poses, dataset_root):
    """
    Build (estimated, GT) camera center correspondences.

    Changed for VGGT: accepts est_poses (V, 4, 4) cam2world numpy array
    instead of a MASt3R scene object.
    """
    est_positions, gt_positions = [], []
    for i, vname in enumerate(view_names):
        c2w_est = est_poses[i] if isinstance(est_poses, np.ndarray) else est_poses[i].cpu().numpy()
        est_positions.append(c2w_est[:3, 3])
        view_dir = os.path.join(dataset_root, vname)
        _, cam2world_gt = load_gt_params(view_dir)
        gt_positions.append(cam2world_gt[:3, 3])
    return np.array(est_positions), np.array(gt_positions)


def build_static_gt_pointcloud(t, view_names, dataset_root,
                               flow_threshold=2.0, use_sam2=True):
    all_pts = []

    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)

        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw * DEPTH_SCALE
        H, W = depth_m.shape

        rgb_dir = os.path.join(view_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            rgb_dir = view_dir

        def _rgb_path(frame_t):
            for ext in (".png", ".jpg", ".jpeg"):
                p = os.path.join(rgb_dir, f"{frame_t:05d}{ext}")
                if os.path.exists(p):
                    return p
            return None

        rgb_t   = _rgb_path(t)
        rgb_adj = _rgb_path(t + 1) or _rgb_path(t - 1)

        static_mask = np.ones((H, W), dtype=bool)
        if rgb_t is not None and rgb_adj is not None:
            if use_sam2:
                from vggt.utils.optical_flow import compute_static_mask
                sam2_mask = compute_static_mask([rgb_t, rgb_adj])
                if sam2_mask is not None:
                    if sam2_mask.shape != (H, W):
                        static_mask = cv2.resize(
                            sam2_mask.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        static_mask = sam2_mask
            else:
                f0 = cv2.imread(rgb_t,   cv2.IMREAD_GRAYSCALE).astype(np.float32)
                f1 = cv2.imread(rgb_adj, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                if f0.shape == f1.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        f0, f1, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                    flow_mask = np.linalg.norm(flow, axis=-1) < flow_threshold
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

from eval_config import CONF_PERCENTILE
def get_static_correspondences(t, view_names, pts3d_list, confs, dataset_root,
                               flow_threshold=2.0,
                               conf_percentile=CONF_PERCENTILE, use_sam2=True,
                               use_static_mask=True):
    """
    Build (estimated, GT) 3-D point correspondences for static regions.

    Changed for VGGT: accepts pts3d_list and confs arrays directly instead of
    a MASt3R scene object.  The caller is responsible for extracting these from
    the model prediction dict and converting to numpy.

    Parameters
    ----------
    pts3d_list : list[np.ndarray] or np.ndarray
        Per-view 3D pointmaps.  Each entry can be (H, W, 3) or (N, 3).
        If a single array of shape (V, H, W, 3), it will be indexed by view.
    confs : list[np.ndarray] or np.ndarray
        Per-view confidence maps.  Each entry can be (H, W) or (N,).

    Strategy — work at native model pointmap resolution to avoid interpolating
    3-D points:

      1. Get pointmap  p3d[h_mod, w_mod, 3]  and conf[h_mod, w_mod].
         These are already at the model's internal resolution — no resize needed.

      2. Downsample GT depth  (H, W) → (h_mod, w_mod)  with INTER_NEAREST so
         depth values are never blended, and scale K accordingly.

      3. Downsample the Farneback static mask the same way.

      4. Backproject the downsampled GT depth at every valid pixel using the
         scaled K  →  GT pointmap at model resolution.

      5. Read the model pointmap at the same pixels  →  estimated pointmap.

      6. The paired (estimated, GT) 3-D points are the correspondences.
    """
    all_est = []
    all_gt  = []

    # Normalise inputs to indexable lists of numpy arrays
    if isinstance(pts3d_list, torch.Tensor):
        pts3d_list = pts3d_list.cpu().numpy()
    if isinstance(confs, torch.Tensor):
        confs = confs.cpu().numpy()
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
        depth_gt  = depth_raw * DEPTH_SCALE
        H, W      = depth_gt.shape

        # ── Model outputs at native model resolution ──────────────────────────
        p3d_model = pts3d_list[i]   # (h_mod, w_mod, 3) or (N, 3)
        conf_mod  = confs[i]        # (h_mod, w_mod)    or (N,)

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
        rgb_dir = os.path.join(view_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            rgb_dir = view_dir

        def _rgb_path(frame_t):
            for ext in (".png", ".jpg", ".jpeg"):
                p = os.path.join(rgb_dir, f"{frame_t:05d}{ext}")
                if os.path.exists(p):
                    return p
            return None

        rgb_t   = _rgb_path(t)
        rgb_adj = _rgb_path(t + 1) or _rgb_path(t - 1)

        static_small = np.ones((h_mod, w_mod), dtype=bool)
        if rgb_t is not None and rgb_adj is not None:
            if use_sam2:
                from vggt.utils.optical_flow import compute_static_mask
                sam2_mask = compute_static_mask([rgb_t, rgb_adj])
                if sam2_mask is not None:
                    static_small = cv2.resize(
                        sam2_mask.astype(np.uint8), (w_mod, h_mod),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                f0 = cv2.imread(rgb_t,   cv2.IMREAD_GRAYSCALE).astype(np.float32)
                f1 = cv2.imread(rgb_adj, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                if f0.shape == f1.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        f0, f1, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                    flow_mask = np.linalg.norm(flow, axis=-1) < flow_threshold
                    static_small = cv2.resize(
                        flow_mask.astype(np.uint8), (w_mod, h_mod),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

        # ── Valid pixel mask (all criteria at model resolution) ────────────────
        valid = (conf_mod   > frame_thr) \
              & (depth_small > 0) \
              & (depth_small < DEPTH_MAX_M)

        if use_static_mask:
            valid &= static_small

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

def get_single_view_correspondences(t, vname, pts3d, conf, dataset_root,
                                     static_mask=None, conf_percentile=CONF_PERCENTILE,
                                     use_static_mask=True):
    """
    Build (estimated, GT) 3-D point correspondences for ONE view.

    Like get_static_correspondences() but for a single view — used when each
    view's pointmap is in a different coordinate frame (VGGT4D per-view
    temporal processing).

    Parameters
    ----------
    t : int — frame index
    vname : str — view folder name (e.g. "view_01")
    pts3d : np.ndarray (H, W, 3) — model pointmap for this view
    conf : np.ndarray (H, W) — confidence map
    dataset_root : str — path to subject root
    static_mask : np.ndarray (H, W) bool, optional
        True = static pixel (from VGGT4D dynamic mask, inverted).
        If None, all pixels are considered static.
    conf_percentile : float

    Returns
    -------
    # Threshold for this frame/view
    # Since this is single-view, we use the local quantile
    """
    thr = np.quantile(conf, 1.0 - conf_percentile)

    view_dir = os.path.join(dataset_root, vname)

    depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
    if not os.path.exists(depth_path):
        return None, None

    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_gt = depth_raw * DEPTH_SCALE
    H_gt, W_gt = depth_gt.shape

    # Recover 2-D spatial shape of model output
    if conf.ndim == 1:
        side = int(np.sqrt(len(conf)))
        h_mod, w_mod = side, side
    else:
        h_mod, w_mod = conf.shape[:2]

    if pts3d.ndim == 2:
        pts3d = pts3d.reshape(h_mod, w_mod, 3)
    conf = conf.reshape(h_mod, w_mod)

    # Scale K to match model resolution
    K, cam2world = load_gt_params(view_dir)
    scale_x = w_mod / W_gt
    scale_y = h_mod / H_gt
    fx_s = K[0, 0] * scale_x
    fy_s = K[1, 1] * scale_y
    cx_s = K[0, 2] * scale_x
    cy_s = K[1, 2] * scale_y

    # Downsample GT depth to model resolution (INTER_NEAREST = no blending)
    depth_small = cv2.resize(depth_gt, (w_mod, h_mod),
                             interpolation=cv2.INTER_NEAREST)

    # Static mask
    if static_mask is None:
        static_small = np.ones((h_mod, w_mod), dtype=bool)
    else:
        if static_mask.shape != (h_mod, w_mod):
            static_small = cv2.resize(
                static_mask.astype(np.uint8), (w_mod, h_mod),
                interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            static_small = static_mask

    # Valid pixel mask
    valid = (conf.reshape(h_mod, w_mod) > thr) \
          & (depth_small > 0) \
          & (depth_small < DEPTH_MAX_M)

    if use_static_mask:
        valid &= static_small

    if not np.any(valid):
        return None, None

    ys, xs = np.where(valid)

    # GT backprojection with scaled K
    z_gt = depth_small[ys, xs]
    pts_cam_gt = np.stack([(xs - cx_s) * z_gt / fx_s,
                           (ys - cy_s) * z_gt / fy_s,
                           z_gt], axis=-1)
    pts_world_gt = (cam2world[:3, :3] @ pts_cam_gt.T).T + cam2world[:3, 3]

    # Model points at the same pixels
    pts_model = pts3d[ys, xs]

    if len(pts_model) < 3:
        return None, None

    return pts_model, pts_world_gt


def build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=1.5, target_hw=None):
    """
    Returns a list of boolean 2D masks (one per view), True where the GT depth
    is valid (> 0) and within depth_max_m.  Optionally resized to target_hw=(H,W)
    to match the MASt3R output pointmap resolution.
    """
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
    return masks