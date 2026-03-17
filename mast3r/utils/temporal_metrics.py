import numpy as np
from scipy.spatial import cKDTree


def compute_l2_error(est_points, gt_points):
    """
    Mean L2 Reconstruction Error
    Measures accuracy relative to ground truth.
    E_t = mean(||p_est - p_gt||_2)
    p_est = aligned estimated point
    p_gt = corresponding ground truth point
    """
    tree_gt = cKDTree(gt_points)
    dists, _ = tree_gt.query(est_points, k=1)
    return np.mean(dists)


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


def compute_temporal_jitter(point_sequence):
    """
    Temporal Jitter
    Measures frame-to-frame instability.
    J_t = mean(||P_est(t+1) - P_est(t)||_2)
    """
    point_sequence = np.asarray(point_sequence)
    diffs = np.linalg.norm(point_sequence[1:] - point_sequence[:-1], axis=-1)
    return np.mean(diffs, axis=-1)


def compute_temporal_variance(point_sequence):
    """
    Temporal Variance
    Measures global temporal stability across the sequence.
    Given point_sequence of shape (T, N, 3).
    variance = var(points_over_time, axis=0)
    temporal_variance = mean(||variance||)
    """
    point_sequence = np.asarray(point_sequence)
    variance = np.var(point_sequence, axis=0)
    norm_variance = np.linalg.norm(variance, axis=-1)
    return np.mean(norm_variance)
