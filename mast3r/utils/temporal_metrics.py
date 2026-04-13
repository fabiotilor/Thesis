import numpy as np
from scipy.spatial import cKDTree


def compute_chamfer_distance(est_points, gt_points):
    """
    Chamfer Distance
    Measures geometric similarity between point clouds.
    CD(P,Q) = mean_{p \in P} min_{q \in Q} ||p-q|| + mean_{q \in Q} min_{p \in P} ||q-p||
    """
    tree_est = cKDTree(est_points)
    tree_gt = cKDTree(gt_points)

    dist_est_to_gt, _ = tree_gt.query(est_points, k=1)
    dist_gt_to_est, _ = tree_est.query(gt_points, k=1)

    return np.mean(dist_est_to_gt) + np.mean(dist_gt_to_est)


def compute_accuracy(est_points, gt_points, tau=0.01):
    """
    Find the nearest neighbor in gt_points for every point in est_points.
    Return the percentage of est_points where the distance is < tau.
    """
    if len(est_points) == 0 or len(gt_points) == 0:
        return np.nan
    tree_gt = cKDTree(gt_points)
    dists, _ = tree_gt.query(est_points, k=1)
    return float(np.mean(dists < tau))


def compute_completeness(est_points, gt_points, tau=0.01):
    """
    Find the nearest neighbor in est_points for every point in gt_points.
    Return the percentage of gt_points where the distance is < tau.
    """
    if len(est_points) == 0 or len(gt_points) == 0:
        return np.nan
    tree_est = cKDTree(est_points)
    dists, _ = tree_est.query(gt_points, k=1)
    return float(np.mean(dists < tau))


def split_points_by_mask(est_points, masks_2d, Ks, R_ts):
    """
    Project the 3D est_points into the 2D image plane of each view.
    masks_2d: list of 2D masks (True=static, False=dynamic)
    Ks: list of 3x3 intrinsics
    R_ts: list of 3x4 or 4x4 world-to-camera transforms

    A point is dynamic if it projects to a dynamic region in ANY view.
    A point is static if it projects to a static region in ALL views it falls into.
    """
    if len(est_points) == 0:
        return est_points, est_points

    num_pts = est_points.shape[0]
    is_dynamic = np.zeros(num_pts, dtype=bool)
    is_visible_in_any = np.zeros(num_pts, dtype=bool)

    pts_homo = np.concatenate([est_points, np.ones((num_pts, 1))], axis=-1)

    for mask_2d, K, R_t in zip(masks_2d, Ks, R_ts):
        if R_t.shape == (4, 4):
            cam_pts = (R_t @ pts_homo.T).T
        elif R_t.shape == (3, 4):
            cam_pts = (R_t @ pts_homo.T).T
        else:
            cam_pts = (R_t @ est_points.T).T

        uvz = (K @ cam_pts[:, :3].T).T
        z = uvz[:, 2]
        z_safe = np.where(z == 0, 1e-6, z)

        u = np.round(uvz[:, 0] / z_safe).astype(int)
        v = np.round(uvz[:, 1] / z_safe).astype(int)

        h, w = mask_2d.shape
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)

        is_visible_in_any |= valid

        # In this view, which points are dynamic?
        # mask is True for static, False for dynamic
        view_dynamic = np.zeros(num_pts, dtype=bool)
        view_dynamic[valid] = ~mask_2d[v[valid], u[valid]]

        is_dynamic |= view_dynamic

    static_pts = est_points[~is_dynamic & is_visible_in_any]
    dynamic_pts = est_points[is_dynamic]

    return static_pts, dynamic_pts


def compute_static_jitter(
        pointmaps: list[np.ndarray],
        static_masks: list[np.ndarray],
        n_anchors: int = 5000,
        seed: int = 42,
) -> dict:
    """
    Measures frame-to-frame instability of the static reconstruction using
    pointmap-based correspondences.

    CALLER NOTE: pointmaps (H, W, 3) must be saved during reconstruction in
    align_reconstruction_umeyama.py by adding 'pointmap': pts3d_list[i] to
    save_dict for each view, or as a stacked array 'pointmaps': np.stack(pts3d_list).
    The static_masks can be reconstructed from the saved 'masks_2d' arrays.
    Umeyama alignment must be applied to the pointmap before passing it here:
      aligned_pointmap = apply_similarity_transform(
          pts3d_list[i].reshape(-1, 3), s, R, tr
      ).reshape(H, W, 3)
    """
    import cv2

    if len(pointmaps) < 2:
        return {'jitter_mean': np.nan, 'jitter_std': np.nan, 'n_anchors': 0, 'n_frames': 0}

    H, W = pointmaps[0].shape[:2]

    # Resize all masks to match the pointmap resolution (H, W) and compute intersection
    intersection_mask = np.ones((H, W), dtype=bool)
    for mask in static_masks:
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        intersection_mask &= mask

    flat_indices = np.where(intersection_mask.flatten())[0]
    print(f"[jitter] Found {len(flat_indices):,} potential static anchors in intersection mask.")

    if len(flat_indices) < 2:
        print("[jitter] [WARN] Fewer than 2 anchors found; skipping jitter computation.")
        return {'jitter_mean': np.nan, 'jitter_std': np.nan, 'n_anchors': 0, 'n_frames': 0}

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(flat_indices, size=min(n_anchors, len(flat_indices)), replace=False)
    print(f"[jitter] Sampled {len(sampled_indices):,} anchors for measurement.")

    # Convert flat indices to (u, v) pairs
    v, u = np.unravel_index(sampled_indices, (H, W))

    # Stack pointmaps for vectorized extraction: (T, H, W, 3)
    stacked_pointmaps = np.stack(pointmaps)

    # Extract trajectories: shape (T, N, 3)
    trajectories = stacked_pointmaps[:, v, u]

    # Compute frame-to-frame displacement: shape (T-1, N)
    displacements = np.linalg.norm(trajectories[1:] - trajectories[:-1], axis=-1)

    jitter_mean = np.mean(displacements)
    per_anchor_mean = np.mean(displacements, axis=0)
    jitter_std = np.std(per_anchor_mean)

    return {
        'jitter_mean': float(jitter_mean),
        'jitter_std': float(jitter_std),
        'n_anchors': int(len(sampled_indices)),
        'n_frames': int(len(pointmaps))
    }
