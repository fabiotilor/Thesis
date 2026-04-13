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
        pointmaps_per_frame: list[np.ndarray],
        masks_per_frame: list[np.ndarray],
        Ks_per_frame: list[np.ndarray] = None,
        R_ts_per_frame: list[np.ndarray] = None,
        n_anchors: int = 5000,
        seed: int = 42,
) -> dict:
    """
    Measure frame-to-frame instability of the static reconstruction using
    multi-view fused pointmap correspondences.

    For each frame the V per-view pointmaps are fused into a single (H, W, 3)
    pointmap via majority-vote static masking and mean-pooling of static-view
    3D positions.  Persistent anchor pixels are sampled from the intersection
    of fused static masks across all frames, and displacement statistics are
    computed over the resulting trajectories.

    Parameters
    ----------
    pointmaps_per_frame : list of np.ndarray
        T arrays, each of shape (V, H, W, 3).  Umeyama-aligned world coords.
    masks_per_frame : list of np.ndarray
        T arrays, each of shape (V, H, W).  True = static pixel.
    Ks_per_frame : list of np.ndarray, optional
        T arrays, each (V, 3, 3).  Intrinsics (reserved for future use).
    R_ts_per_frame : list of np.ndarray, optional
        T arrays, each (V, 4, 4).  Extrinsics (reserved for future use).
    n_anchors : int
        Number of anchor pixels to sample from the intersection mask.
    seed : int
        Random seed for reproducible anchor sampling.

    Returns
    -------
    dict
        jitter_mean      : mean frame-to-frame displacement (metres)
        jitter_std       : std of per-anchor mean displacement
        jitter_p95       : 95th percentile of per-anchor mean displacement
        jitter_max       : max per-anchor mean displacement
        drift_mean       : mean first-to-last-frame displacement per anchor
        hf_jitter        : mean second-order temporal acceleration (nan if T < 3)
        per_frame_jitter : (T-1,) mean displacement per frame transition
        n_anchors        : actual anchors used
        n_frames         : T

    # CALLER NOTE: pointmaps_per_frame contains (V, H, W, 3) arrays — the full
    # multi-view pointmaps saved during reconstruction.  Each view's pointmap
    # must already be Umeyama-aligned into the GT coordinate frame.  The
    # masks_per_frame are the per-view flow masks (V, H, W) from 'masks_2d'.
    # Ks_per_frame and R_ts_per_frame are the per-view camera parameters from
    # 'Ks' and 'R_ts' respectively.
    """
    import cv2

    nan_result = {
        'jitter_mean': np.nan, 'jitter_std': np.nan, 'jitter_p95': np.nan,
        'jitter_max': np.nan, 'drift_mean': np.nan, 'hf_jitter': np.nan,
        'per_frame_jitter': np.array([]), 'n_anchors': 0, 'n_frames': 0,
    }

    T = len(pointmaps_per_frame)
    if T < 2:
        return nan_result

    # Determine spatial resolution from first frame, first view
    first_pm = pointmaps_per_frame[0]  # (V, H, W, 3)
    V, H, W, _ = first_pm.shape

    # ── Fuse views per frame ─────────────────────────────────────────────
    fused_pointmaps = []  # T x (H, W, 3)
    fused_masks = []      # T x (H, W) bool

    for t in range(T):
        pm_t = pointmaps_per_frame[t]   # (V, H, W, 3)
        mk_t = masks_per_frame[t]       # (V, H', W') — may need resize

        # Resize masks to (V, H, W) if needed
        resized_masks = np.empty((V, H, W), dtype=bool)
        for vi in range(V):
            m = mk_t[vi]
            if m.shape != (H, W):
                m = cv2.resize(
                    m.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            resized_masks[vi] = m

        # Majority-vote static mask: static in more than half of views
        static_count = resized_masks.sum(axis=0)  # (H, W)
        fused_mask_t = static_count > (V / 2.0)   # (H, W) bool

        # Mean of 3D positions from views where pixel is static
        # Use masked operations: set non-static entries to NaN, then nanmean
        pm_masked = pm_t.copy()  # (V, H, W, 3)
        mask_expanded = resized_masks[:, :, :, np.newaxis]  # (V, H, W, 1)
        pm_masked[~np.broadcast_to(mask_expanded, pm_masked.shape)] = np.nan

        with np.errstate(all='ignore'):
            fused_pm_t = np.nanmean(pm_masked, axis=0)  # (H, W, 3)

        # Pixels where no view is static → NaN; mark as invalid
        all_nan = np.all(np.isnan(fused_pm_t), axis=-1)  # (H, W)
        fused_mask_t = fused_mask_t & (~all_nan)

        fused_pointmaps.append(fused_pm_t)
        fused_masks.append(fused_mask_t)

    # ── Anchor intersection mask ─────────────────────────────────────────
    anchor_mask = np.ones((H, W), dtype=bool)
    for t in range(T):
        anchor_mask &= fused_masks[t]
        # Also require non-NaN in all 3 coords
        anchor_mask &= ~np.any(np.isnan(fused_pointmaps[t]), axis=-1)

    flat_indices = np.where(anchor_mask.ravel())[0]
    print(f"[jitter] Fused {V} views -> {len(flat_indices):,} potential anchors "
          f"in intersection mask ({H}x{W}).")

    if len(flat_indices) < 2:
        print("[jitter] [WARN] Fewer than 2 anchors; skipping jitter computation.")
        return nan_result

    # ── Sample anchors ───────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    n_sample = min(n_anchors, len(flat_indices))
    sampled_indices = rng.choice(flat_indices, size=n_sample, replace=False)
    print(f"[jitter] Sampled {n_sample:,} anchors for measurement.")

    v_idx, u_idx = np.unravel_index(sampled_indices, (H, W))

    # ── Extract trajectories: (T, N, 3) ─────────────────────────────────
    stacked = np.stack(fused_pointmaps)  # (T, H, W, 3)
    trajectories = stacked[:, v_idx, u_idx]  # (T, N, 3)

    # ── Frame-to-frame displacement: (T-1, N) ───────────────────────────
    displacements = np.linalg.norm(
        trajectories[1:] - trajectories[:-1], axis=-1,
    )

    per_anchor_mean = np.mean(displacements, axis=0)  # (N,)
    jitter_mean = float(np.mean(displacements))
    jitter_std = float(np.std(per_anchor_mean))
    jitter_p95 = float(np.percentile(per_anchor_mean, 95))
    jitter_max = float(np.max(per_anchor_mean))

    # Per-frame jitter: mean displacement per frame transition (T-1,)
    per_frame_jitter = np.mean(displacements, axis=1)  # (T-1,)

    # Drift: displacement between first and last frame per anchor
    drift = np.linalg.norm(trajectories[-1] - trajectories[0], axis=-1)  # (N,)
    drift_mean = float(np.mean(drift))

    # High-frequency jitter: second-order temporal acceleration
    if T >= 3:
        accel = np.linalg.norm(
            trajectories[2:] - 2 * trajectories[1:-1] + trajectories[:-2],
            axis=-1,
        )  # (T-2, N)
        hf_jitter = float(np.mean(accel))
    else:
        hf_jitter = np.nan

    return {
        'jitter_mean': jitter_mean,
        'jitter_std': jitter_std,
        'jitter_p95': jitter_p95,
        'jitter_max': jitter_max,
        'drift_mean': drift_mean,
        'hf_jitter': hf_jitter,
        'per_frame_jitter': per_frame_jitter,
        'n_anchors': int(n_sample),
        'n_frames': int(T),
    }
