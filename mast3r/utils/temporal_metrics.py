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
    dynamic_pts = est_points[is_dynamic & is_visible_in_any]

    return static_pts, dynamic_pts


def compute_static_jitter(
        pointmaps_per_frame: list[np.ndarray],
        masks_per_frame: list[np.ndarray],
        Ks_per_frame: list[np.ndarray] = None,
        R_ts_per_frame: list[np.ndarray] = None,
        validity_masks_per_frame: list[np.ndarray] = None,
        confidences_per_frame: list[np.ndarray] = None,
        conf_percentile: float = 0.5,
        n_anchors: int = 5000,
        seed: int = 42,
) -> dict:
    """
    Measure frame-to-frame instability of static reconstruction samples using
    per-view persistent pointmap anchors.

    Anchors are fixed pixel locations in fixed camera views, i.e. each anchor is
    identified by (view, y, x).  A candidate anchor is kept only if it is static,
    high-confidence when confidences are provided, finite/non-zero, and
    optionally GT-valid in every frame.  Sampling from the union of all per-view
    persistent anchors naturally allocates anchors proportionally to the number
    of valid persistent locations in each view.

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
    validity_masks_per_frame : list of np.ndarray, optional
        T arrays, each of shape (V, H, W).  True = GT-valid / evaluable pixel.
    confidences_per_frame : list of np.ndarray, optional
        T arrays, each of shape (V, H, W).  Model confidence values.
    conf_percentile : float
        Fraction of points to retain per frame/view when confidences are given.
    n_anchors : int
        Number of (view, pixel) anchors to sample from the persistent set.
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

    # CALLER NOTE: pointmaps_per_frame contains (V, H, W, 3) arrays.  Each
    # view's pointmap must already be Umeyama-aligned into the GT coordinate
    # frame.  The masks_per_frame are the per-view static masks from 'masks_2d'.
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
    if validity_masks_per_frame is not None and len(validity_masks_per_frame) != T:
        print("[jitter] [WARN] Incomplete validity masks; falling back to static masks only.")
        validity_masks_per_frame = None
    if confidences_per_frame is not None and len(confidences_per_frame) != T:
        print("[jitter] [WARN] Incomplete confidence maps; falling back to no confidence filtering.")
        confidences_per_frame = None

    first_pm = pointmaps_per_frame[0]  # (V, H, W, 3)
    V, H, W, _ = first_pm.shape

    def _resize_mask_stack(mask_stack, fill=True):
        if mask_stack is None:
            return np.full((V, H, W), fill, dtype=bool)
        resized = np.empty((V, H, W), dtype=bool)
        for vi in range(V):
            m = mask_stack[vi]
            if m is None:
                resized[vi] = fill
                continue
            if m.shape != (H, W):
                m = cv2.resize(
                    m.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            resized[vi] = m
        return resized

    def _resize_conf_stack(conf_stack):
        if conf_stack is None:
            return None
        resized = np.empty((V, H, W), dtype=np.float32)
        for vi in range(V):
            c = conf_stack[vi]
            if c.shape != (H, W):
                c = cv2.resize(
                    c.astype(np.float32), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                )
            resized[vi] = c
        return resized

    valid_masks = []  # T x (V, H, W)
    for t in range(T):
        pm_t = pointmaps_per_frame[t]  # (V, H, W, 3)
        static_mask = _resize_mask_stack(masks_per_frame[t], fill=False)
        if validity_masks_per_frame is not None:
            gt_valid = _resize_mask_stack(validity_masks_per_frame[t], fill=True)
        else:
            gt_valid = np.ones((V, H, W), dtype=bool)

        if confidences_per_frame is not None:
            conf_t = _resize_conf_stack(confidences_per_frame[t])
            conf_mask = np.zeros((V, H, W), dtype=bool)
            for vi in range(V):
                c = conf_t[vi]
                finite_conf = np.isfinite(c)
                if np.any(finite_conf):
                    thr = np.percentile(c[finite_conf], 100 * (1 - conf_percentile))
                    conf_mask[vi] = finite_conf & (c > thr)
        else:
            conf_mask = np.ones((V, H, W), dtype=bool)

        finite = np.isfinite(pm_t).all(axis=-1)
        nonzero = np.linalg.norm(pm_t, axis=-1) > 1e-8
        valid_masks.append(static_mask & gt_valid & conf_mask & finite & nonzero)

    anchor_mask = np.ones((V, H, W), dtype=bool)
    for valid_t in valid_masks:
        anchor_mask &= valid_t

    flat_indices = np.where(anchor_mask.ravel())[0]
    per_view_counts = anchor_mask.reshape(V, -1).sum(axis=1)
    print(f"[jitter] Per-view persistent anchors: "
          f"{', '.join(str(int(c)) for c in per_view_counts)} "
          f"(total={len(flat_indices):,}, resolution={H}x{W}).")

    if len(flat_indices) < 2:
        print("[jitter] [WARN] Fewer than 2 anchors; skipping jitter computation.")
        return nan_result

    rng = np.random.default_rng(seed)
    n_sample = min(n_anchors, len(flat_indices))
    sampled_indices = rng.choice(flat_indices, size=n_sample, replace=False)
    print(f"[jitter] Sampled {n_sample:,} anchors for measurement.")

    view_idx, y_idx, x_idx = np.unravel_index(sampled_indices, (V, H, W))

    stacked = np.stack(pointmaps_per_frame)  # (T, V, H, W, 3)
    trajectories = stacked[:, view_idx, y_idx, x_idx]  # (T, N, 3)

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
        'n_potential_anchors': int(len(flat_indices)),
        'per_view_anchor_counts': per_view_counts.astype(int),
    }


def rotation_error_deg(R_est, R_gt):
    """
    Geodesic distance in degrees: arccos((tr(R_est^T R_gt) - 1) / 2)
    """
    R_diff = R_est.T @ R_gt
    tr = np.trace(R_diff)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))


def intrinsic_errors(K_est, K_gt):
    """
    returns focal_err, pp_err
    """
    f_est = (K_est[0, 0] + K_est[1, 1]) / 2.0
    f_gt = (K_gt[0, 0] + K_gt[1, 1]) / 2.0
    focal_err = float(np.abs(f_est - f_gt) / f_gt) if f_gt != 0 else np.nan

    pp_est = K_est[:2, 2]
    pp_gt = K_gt[:2, 2]
    pp_err = float(np.linalg.norm(pp_est - pp_gt))

    return focal_err, pp_err


def ate_rms(centers_est, centers_gt):
    errs = np.linalg.norm(centers_est - centers_gt, axis=1)
    return float(np.sqrt(np.mean(errs ** 2)))


def compute_camera_metrics(poses_est, poses_gt, intrinsics_est, intrinsics_gt, s, R, t):
    """
    poses_est: (V, 4, 4) estimated cam2world transforms
    poses_gt: (V, 4, 4) ground truth cam2world transforms
    intrinsics_est: (V, 3, 3)
    intrinsics_gt: (V, 3, 3)
    s, R, t: Umeyama alignment parameters
    """
    if len(poses_est) == 0:
        return {
            'ate': np.nan,
            'rpe': np.nan,
            'rot_error': np.nan,
            'focal_error': np.nan,
            'pp_error': np.nan
        }

    V = poses_est.shape[0]

    C_est = poses_est[:, :3, 3]
    R_est = poses_est[:, :3, :3]

    C_gt = poses_gt[:, :3, 3]
    R_gt = poses_gt[:, :3, :3]

    # C_world = s * (R_align * C_est + t_align)
    C_est_aligned = s * ((R @ C_est.T).T + t)

    R_est_aligned = np.zeros_like(R_est)
    for i in range(V):
        R_est_aligned[i] = R @ R_est[i]

    ate = ate_rms(C_est_aligned, C_gt)

    rpe_errors = []
    if V >= 2:
        for i in range(V):
            for j in range(i + 1, V):
                rel_t_est = C_est_aligned[i] - C_est_aligned[j]
                rel_t_gt = C_gt[i] - C_gt[j]
                rpe_errors.append(np.linalg.norm(rel_t_est - rel_t_gt))
        rpe = float(np.sqrt(np.mean(np.array(rpe_errors) ** 2)))
    else:
        rpe = np.nan

    rot_errs = []
    focal_errs = []
    pp_errs = []

    for i in range(V):
        rot_errs.append(rotation_error_deg(R_est_aligned[i], R_gt[i]))
        ferr, pperr = intrinsic_errors(intrinsics_est[i], intrinsics_gt[i])
        focal_errs.append(ferr)
        pp_errs.append(pperr)

    return {
        'ate': ate,
        'rpe': rpe,
        'rot_error': float(np.nanmean(rot_errs)),
        'focal_error': float(np.nanmean(focal_errs)),
        'pp_error': float(np.nanmean(pp_errs))
    }
