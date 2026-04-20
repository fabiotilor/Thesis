import numpy as np
import cv2
import os

# Global cache for SAM2 model to avoid reloading for every frame
_SAM2_CACHE = {
    "model": None,
    "predictor": None
}


def compute_flow_sam2(frames: list[np.ndarray]) -> np.ndarray:
    """
    Returns a motion / static mask proxy using SAM2 segmentation.
    This extracts coarse inter-frame frame differences to derive dynamic regions,
    and then feeds points into SAM2 Image Predictor for robust segmentation.
    Returns:
        mask (np.ndarray): Boolean mask (H, W) where True = static, False = dynamic.
    """
    global _SAM2_CACHE

    import sys
    import os
    sam2_path = os.path.abspath('sam2')

    # We lazily import SAM2 to avoid requiring it in all workflows
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except (ImportError, RuntimeError):
        if os.path.isdir(sam2_path) and sam2_path not in sys.path:
            sys.path.insert(0, sam2_path)

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            print(f"[ERROR] SAM2 import failed: {e}")
            print("[HINT] You may need to install SAM2: cd sam2 && pip install -e .")
            raise e
        except RuntimeError as e:
            if "shadowed by the repository name" in str(e):
                print(f"[ERROR] SAM2 Shadowing Error: {e}")
                print("[HINT] Try properly installing SAM2: cd sam2 && pip install -e .")
            raise e

    # Use cached predictor if available
    if _SAM2_CACHE["predictor"] is not None:
        predictor = _SAM2_CACHE["predictor"]
    else:
        # Initialize model
        checkpoint = "./sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(checkpoint):
            print(f"[WARN] SAM2 checkpoint not found at {checkpoint}! Falling back to static mask.")
            H, W = frames[0].shape[:2]
            return np.ones((H, W), dtype=bool)

        print(f"[INFO] Initializing SAM2 model from {checkpoint}...")

        # Explicit hydra context locally to ensure GlobalHydra is fully activated
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        if not GlobalHydra.instance().is_initialized():
            # config dir is sam2/sam2 (parent of configs/)
            # this allows build_sam2 to find "configs/sam2.1/..."
            conf_dir = os.path.abspath(os.path.join(sam2_path, "sam2"))
            with initialize_config_dir(config_dir=conf_dir, version_base="1.2"):
                sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        else:
            sam2_model = build_sam2(model_cfg, checkpoint, device=device)

        predictor = SAM2ImagePredictor(sam2_model)
        _SAM2_CACHE["model"] = sam2_model
        _SAM2_CACHE["predictor"] = predictor

    H, W = frames[0].shape[:2]
    static_mask = np.ones((H, W), dtype=bool)

    for i in range(len(frames) - 1):
        f0 = frames[i]
        f1 = frames[i + 1]

        # 1. Coarse motion detection via absdiff
        # Convert to gray if needed
        gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY) if len(f0.shape) == 3 else f0
        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if len(f1.shape) == 3 else f1

        diff = cv2.absdiff(gray0, gray1)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Median blur to remove noise
        thresh = cv2.medianBlur(thresh, 5)

        # 2. Find contours to get bounding boxes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter extremely small and extremely large noisy diffs
            if 100 < area < (H * W * 0.8):
                x, y, w, h = cv2.boundingRect(cnt)
                # SAM expects boxes in XYXY format
                boxes.append([x, y, x + w, y + h])

        if not boxes:
            # Entire frame is static
            continue

        boxes_np = np.array(boxes)

        # 3. Predict accurate segmentation via SAM2
        # Using frame0 + frame1 ?
        # Typically run on the second frame to segment the moved object
        img_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB) if len(f1.shape) == 3 else cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB)
        predictor.set_image(img_rgb)

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_np,
            multimask_output=False,
        )

        if masks.size > 0:
            # Combine all predicted masks and invert to get static
            # Handle different possible return shapes from SAM2 (N, 1, H, W) or (1, H, W) etc.
            if masks.ndim == 4:
                # Expected (N, 1, H, W) -> (N, H, W)
                masks = masks.squeeze(1)

            # Now masks is (N, H, W). Sum along axis 0 to union all box masks
            dynamic = masks.any(axis=0) if masks.ndim == 3 else masks
            static_mask &= (~dynamic)

    return static_mask


def compute_static_mask(rgb_paths, flow_threshold=1.0, method="farneback"):
    """
    Returns a boolean mask (H, W) where True = static across all frame transitions.
    Arguments:
        rgb_paths: list of paths to images
        flow_threshold: float, used only for farneback
        method: "farneback" or "sam2"
    """
    if method == "sam2":
        # Load all frames
        frames = []
        for p in rgb_paths:
            img = cv2.imread(p)
            if img is not None:
                frames.append(img)
        if len(frames) < 2:
            return None
        return compute_flow_sam2(frames)

    # Fallback to standard Farneback
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