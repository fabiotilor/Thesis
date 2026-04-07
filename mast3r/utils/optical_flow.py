import numpy as np
import cv2

def compute_static_mask(rgb_paths, flow_threshold=1.0):
    """
    Returns a boolean mask (H, W) where True = static across all frame transitions.
    """
    H, W = cv2.imread(rgb_paths[0], cv2.IMREAD_GRAYSCALE).shape
    static = np.ones((H, W), dtype=bool)

    for i in range(len(rgb_paths) - 1):
        f0 = cv2.imread(rgb_paths[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        f1 = cv2.imread(rgb_paths[i + 1], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # Use Farneback for efficiency and strictness
        flow = cv2.calcOpticalFlowFarneback(
            f0, f1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        static &= (magnitude < flow_threshold)
    return static