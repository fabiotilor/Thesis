import numpy as np
import os
import cv2
from .camera_utils import get_rgb_path, remove_outliers
from eval_config import CONF_PERCENTILE, CLEAN_DEPTH

DEPTH_MAX_M = 1.5
DEPTH_SCALE = 0.001  # mm → metres


def load_gt_params(view_dir, dataset_type="dex-ycb"):
    if dataset_type == "dex-ycb":
        data = np.load(os.path.join(view_dir, "intrinsics_extrinsics.npz"))
        K = data['intrinsics'].astype(np.float64)[:3, :3]
        cam2world = np.linalg.inv(data['extrinsics'].astype(np.float64))
        return K, cam2world
    elif dataset_type == "hi4d":
        # For Hi4D, view_dir is usually .../pairXX/actionXX/images/ID or .../pairXX/actionXX/ID
        # The cameras are in .../pairXX/actionXX/cameras/rgb_cameras.npz
        action_dir = os.path.dirname(view_dir)
        if os.path.basename(action_dir) == "images":
            action_dir = os.path.dirname(action_dir)

        cam_id = os.path.basename(view_dir)
        cam_path = os.path.join(action_dir, "cameras", "rgb_cameras.npz")

        data = np.load(cam_path)
        print(f"  [DEBUG] Hi4D camera file keys: {list(data.keys())}")

        # Try different possible key names for camera IDs
        if 'ids' in data:
            ids = list(data['ids'])
        elif 'cam_ids' in data:
            ids = list(data['cam_ids'])
        else:
            print(f"  [ERROR] No camera IDs found in Hi4D camera file: {cam_path}")
            raise ValueError(f"No camera IDs found in {cam_path}")

        # Try different possible key names for intrinsics
        if 'intrinsics' in data:
            intrinsics = data['intrinsics']
        elif 'intrinsic' in data:
            intrinsics = data['intrinsic']
        elif 'K' in data:
            intrinsics = data['K']
        else:
            print(f"  [ERROR] No intrinsics found in Hi4D camera file: {cam_path}")
            raise ValueError(f"No intrinsics found in {cam_path}")

        # Try different possible key names for extrinsics
        if 'extrinsics' in data:
            extrinsics = data['extrinsics']
        elif 'extrinsic' in data:
            extrinsics = data['extrinsic']
        elif 'poses' in data:
            extrinsics = data['poses']
        else:
            print(f"  [ERROR] No extrinsics found in Hi4D camera file: {cam_path}")
            raise ValueError(f"No extrinsics found in {cam_path}")

        try:
            idx = ids.index(int(cam_id))
        except ValueError:
            try:
                idx = ids.index(str(cam_id))
            except ValueError:
                print(f"  [ERROR] Camera ID {cam_id} not found in camera file. Available IDs: {ids}")
                raise ValueError(f"Camera ID {cam_id} not found in {cam_path}")

        K = intrinsics[idx].astype(np.float64)
        ext = extrinsics[idx].astype(np.float64)  # 3x4 world-to-cam
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
    cx, cy = K[0, 2], K[1, 2]
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = (depth_m > 0) & (depth_m < DEPTH_MAX_M)
    z = depth_m[mask]
    pts = np.stack([(u[mask] - cx) * z / fx, (v[mask] - cy) * z / fy, z], axis=-1)
    return pts, mask


def _load_hi4d_mesh_gt(dataset_root, t):
    """Load raw mesh scan from frames_vis as GT pointcloud for Hi4D."""
    import pickle
    pkl_path = os.path.join(dataset_root, "frames_vis", f"mesh-f{t:05d}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            mesh = pickle.load(f)
        return mesh['vertices']  # (42930, 3) float32
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


def get_camera_correspondences(t, view_names, scene, dataset_root, dataset_type="dex-ycb"):
    from dust3r.utils.device import to_numpy
    est_positions, gt_positions = [], []
    im_poses = scene.get_im_poses()
    for i, vname in enumerate(view_names):
        c2w_est = to_numpy(im_poses[i])
        est_positions.append(c2w_est[:3, 3])
        if dataset_type == "hi4d":
            view_dir = os.path.join(dataset_root, "images", vname)
        else:
            view_dir = os.path.join(dataset_root, vname)
        _, cam2world_gt = load_gt_params(view_dir, dataset_type=dataset_type)
        gt_positions.append(cam2world_gt[:3, 3])
    return np.array(est_positions), np.array(gt_positions)


def _load_hi4d_seg_mask(dataset_root, cam_id, t):
    """Load the person segmentation mask for a Hi4D camera+frame."""
    # Extract subject/pair from dataset_root path
    # dataset_root should be like: /path/to/hi4d/pair09/talk09
    path_parts = dataset_root.split(os.sep)
    hi4d_idx = -1
    for i, part in enumerate(path_parts):
        if part == "hi4d":
            hi4d_idx = i
            break

    if hi4d_idx >= 0 and len(path_parts) > hi4d_idx + 2:
        # We have hi4d/pairXX/actionXX structure
        pair_name = path_parts[hi4d_idx + 1]  # e.g., "pair09"
        action_name = path_parts[hi4d_idx + 2]  # e.g., "talk09"

        # Build the correct path: hi4d/pairXX/actionXX/seg/img_seg_mask/cam_id/all/t:06d.png
        hi4d_root = os.sep.join(path_parts[:hi4d_idx + 1])  # go to hi4d root
        mask_path = os.path.join(
            hi4d_root, pair_name, action_name, "seg", "img_seg_mask", str(cam_id), "all", f"{t:06d}.png"
        )
        print(f"  [DEBUG] Mask path for {cam_id} frame {t}: {mask_path}")
    else:
        # Fallback to old logic if path structure is unexpected
        mask_path = os.path.join(dataset_root, "seg", "img_seg_mask", str(cam_id), f"{t:06d}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(dataset_root, "seg", "img_seg_mask", str(cam_id), "all", f"{t:06d}.png")

    if os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return m > 0

    print(f"  [WARN] Hi4D segmentation mask NOT found for {cam_id} at t={t}: {mask_path}")
    return None


def build_static_gt_pointcloud(t, view_names, dataset_root, precomputed_masks=None, dataset_type="dex-ycb"):
    if dataset_type == "hi4d":
        # Hi4D has no static background in GT scans (person meshes only)
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

        rgb_t = get_rgb_path(view_dir, t)
        rgb_adj = get_rgb_path(view_dir, t + 1) or get_rgb_path(view_dir, t - 1)

        if precomputed_masks is not None and vname in precomputed_masks and precomputed_masks[vname] is not None:
            static_mask = precomputed_masks[vname]
            if static_mask.shape != (H, W):
                static_mask = cv2.resize(static_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(
                    bool)
        else:
            static_mask = np.ones((H, W), dtype=bool)
            if rgb_t is not None and rgb_adj is not None:
                from mast3r.utils.optical_flow import compute_flow_sam2
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
        if DEPTH_MAX_M is not None:
            keep &= (depth_m < DEPTH_MAX_M)

        ys, xs = np.where(keep)
        z = depth_m[ys, xs]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=-1)
        pts_world = (cam2world[:3, :3] @ pts_cam.T).T + cam2world[:3, 3]
        all_pts.append(pts_world)

    return np.concatenate(all_pts, axis=0) if all_pts else None


from eval_config import CONF_PERCENTILE


def get_static_correspondences(t, view_names, scene, dataset_root, conf_percentile=CONF_PERCENTILE,
                               precomputed_masks=None, use_static_mask=True, dataset_type="dex-ycb",
                               return_per_view=False):
    from dust3r.utils.device import to_numpy
    if dataset_type == "hi4d":
        return _get_correspondences_hi4d(t, view_names, scene, dataset_root, conf_percentile=conf_percentile,
                                         return_per_view=return_per_view)

    all_est = []
    all_gt = []
    per_view_dict = {}

    pts3d_list, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))

    for i, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, vname)

        # ── GT depth (original sensor resolution) ─────────────────────────────
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_gt = depth_raw * DEPTH_SCALE
        H, W = depth_gt.shape

        # ── MASt3R outputs at native model resolution ──────────────────────────
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
        rgb_t = get_rgb_path(view_dir, t)
        rgb_adj = get_rgb_path(view_dir, t + 1) or get_rgb_path(view_dir, t - 1)

        if precomputed_masks is not None and vname in precomputed_masks and precomputed_masks[vname] is not None:
            static_small = cv2.resize(precomputed_masks[vname].astype(np.uint8), (w_mod, h_mod),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            static_small = np.ones((h_mod, w_mod), dtype=bool)
            if rgb_t is not None and rgb_adj is not None:
                from mast3r.utils.optical_flow import compute_flow_sam2
                f0 = cv2.imread(rgb_t)
                f1 = cv2.imread(rgb_adj)
                if f0 is not None and f1 is not None and f0.shape == f1.shape:
                    flow_mask = compute_flow_sam2([f0, f1])
                    static_small = cv2.resize(
                        flow_mask.astype(np.uint8), (w_mod, h_mod),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

        # ── Valid pixel mask (all criteria at model resolution) ────────────────
        thr = np.percentile(conf_mod, 100 * (1 - conf_percentile))
        valid = (conf_mod > thr) \
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
        per_view_dict[vname] = (pts_model, pts_world_gt)

    if return_per_view:
        return per_view_dict

    if not all_est:
        return None, None

    return np.concatenate(all_est, axis=0), np.concatenate(all_gt, axis=0)


def _get_correspondences_hi4d(t, view_names, scene, dataset_root, conf_percentile=CONF_PERCENTILE,
                              return_per_view=False):
    """
    Hi4D specific: Projects GT mesh into each view and finds correspondences.
    """
    from dust3r.utils.device import to_numpy
    from .camera_utils import get_rgb_path, remove_outliers
    # 1. Load GT points (meshes)
    gt_pts = _load_hi4d_mesh_gt(dataset_root, t)
    if gt_pts is None: return None, None

    from eval_config import CLEAN_DEPTH
    pts3d_list, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=CLEAN_DEPTH))

    src_pts, dst_pts = [], []
    per_view_dict = {}

    for i, vname in enumerate(view_names):
        view_dir = os.path.join(dataset_root, "images", vname)
        pts = pts3d_list[i].reshape(-1, 3)
        conf = confs[i].ravel()

        thr = np.percentile(conf, 100 * (1 - conf_percentile))
        valid = conf > thr

        h_mod, w_mod = confs[i].shape[:2]

        # Load GT params for this view
        K, cam2world = load_gt_params(view_dir, dataset_type="hi4d")
        conf = confs[i]

        if conf.ndim == 1:
            side = int(np.sqrt(len(conf)))
            h_mod, w_mod = side, side
        else:
            h_mod, w_mod = conf.shape

        if pts.ndim == 2: pts = pts.reshape(h_mod, w_mod, 3)
        conf = conf.reshape(h_mod, w_mod)

        seg_mask = _load_hi4d_seg_mask(dataset_root, vname, t)
        seg_small = cv2.resize(seg_mask.astype(np.uint8), (w_mod, h_mod), interpolation=cv2.INTER_NEAREST).astype(
            bool) if seg_mask is not None else np.ones((h_mod, w_mod), dtype=bool)

        valid = (conf > thr) & seg_small
        if not np.any(valid): continue

        # Determine image size for K scaling
        rgb_path = get_rgb_path(view_dir, t)
        if rgb_path:
            img = cv2.imread(rgb_path)
            H_orig, W_orig = img.shape[:2]
        else:
            H_orig, W_orig = 940, 1280

        K_scaled = K.copy()
        K_scaled[0] *= (w_mod / W_orig)
        K_scaled[1] *= (h_mod / H_orig)

        world2cam = np.linalg.inv(cam2world)
        pts_cam_gt = (world2cam[:3, :3] @ gt_pts.T).T + world2cam[:3, 3]
        valid_z = pts_cam_gt[:, 2] > 0
        pts_cam_gt = pts_cam_gt[valid_z]
        gt_pts_valid = gt_pts[valid_z]

        if len(pts_cam_gt) == 0: continue

        uvz = (K_scaled @ pts_cam_gt.T).T
        u = np.round(uvz[:, 0] / uvz[:, 2]).astype(int)
        v = np.round(uvz[:, 1] / uvz[:, 2]).astype(int)

        depth_buffer = np.full((h_mod, w_mod), np.inf)
        index_buffer = np.full((h_mod, w_mod), -1, dtype=int)

        in_bounds = (u >= 0) & (u < w_mod) & (v >= 0) & (v < h_mod)
        for ui, vi, zi, idxi in zip(u[in_bounds], v[in_bounds], pts_cam_gt[in_bounds, 2], np.where(in_bounds)[0]):
            if zi < depth_buffer[vi, ui]:
                depth_buffer[vi, ui] = zi
                index_buffer[vi, ui] = idxi

        ys, xs = np.where(valid & (index_buffer != -1))
        if len(ys) >= 3:
            raw_src = pts[ys, xs]
            raw_dst = gt_pts_valid[index_buffer[ys, xs]]

            # Use SOR to clean correspondences before Umeyama
            # Apply to source points (estimated)
            clean_src, sor_mask = remove_outliers(raw_src, nb_neighbors=15, std_ratio=0.8, return_mask=True)
            clean_dst = raw_dst[sor_mask]

            src_pts.append(clean_src)
            dst_pts.append(clean_dst)
            per_view_dict[vname] = (clean_src, clean_dst)

    if return_per_view:
        return per_view_dict

    if not src_pts: return None, None
    return np.concatenate(src_pts, axis=0), np.concatenate(dst_pts, axis=0)


def build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=1.5, target_hw=None, cache=None,
                            dataset_type="dex-ycb"):
    """
    Returns a list of boolean 2D masks (one per view), True where the GT depth
    is valid (> 0) and within depth_max_m.  Optionally resized to target_hw=(H,W)
    to match the MASt3R output pointmap resolution.
    """
    # Check cache for this specific frame
    if cache is not None and t in cache:
        return cache[t]

    masks = []
    if dataset_type == "hi4d":
        for vname in view_names:
            m = _load_hi4d_seg_mask(dataset_root, vname, t)
            if m is not None and target_hw is not None:
                m = cv2.resize(m.astype(np.uint8), (target_hw[1], target_hw[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            masks.append(m)
        if cache is not None: cache[t] = masks
        return masks

    for vname in view_names:
        view_dir = os.path.join(dataset_root, vname)
        depth_path = os.path.join(view_dir, "depth", f"{t:05d}.png")

        if not os.path.exists(depth_path):
            masks.append(None)
            continue

        # DexYCB depths are uint16, stored in millimetres
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