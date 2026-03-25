from scipy.spatial import cKDTree
import numpy as np


def estimate_similarity_transform(source_points, target_points):
    """
    Estimate the similarity transform (scale, rotation, translation) that aligns
    source_points to target_points using the Umeyama algorithm.

    T(p) = scale * rotation @ p + translation

    Args:
        source_points (np.ndarray): (N, 3) source point cloud.
        target_points (np.ndarray): (N, 3) target point cloud.

    Returns:
        scale (float)
        rotation (np.ndarray): (3, 3)
        translation (np.ndarray): (3,)
    """
    assert source_points.shape == target_points.shape
    N, m = source_points.shape

    # 1. Centering
    mu_s = source_points.mean(axis=0)
    mu_t = target_points.mean(axis=0)

    s_centered = source_points - mu_s
    t_centered = target_points - mu_t

    # 2. Variance
    var_s = (s_centered ** 2).sum() / N

    # 3. Covariance matrix
    # K = 1/N * target^T * source
    K = (t_centered.T @ s_centered) / N

    # 4. SVD
    U, D, Vt = np.linalg.svd(K)

    # 5. Rotation
    # R = U S V^T (but we need to be careful with reflections)
    # The Umeyama paper uses S = diag(1, 1, det(U)det(V))
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1

    R = U @ S @ Vt

    # 6. Scale
    scale = np.sum(np.diag(D) * S) / var_s if var_s > 0 else 1.0

    # 7. Translation
    translation = mu_t - scale * (R @ mu_s)

    return scale, R, translation


def apply_similarity_transform(points, scale, rotation, translation):
    """
    Apply a similarity transform to a point cloud.

    Args:
        points (np.ndarray): (N, 3)
        scale (float)
        rotation (np.ndarray): (3, 3)
        translation (np.ndarray): (3,)

    Returns:
        transformed_points (np.ndarray): (N, 3)
    """
    return scale * (points @ rotation.T) + translation
