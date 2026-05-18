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
        except Exception as e:
            print(f"[ERROR] SAM2 import failed: {e}")
            print("[HINT] You may need to install SAM2: cd sam2 && pip install -e .")
            raise e

    # Use cached predictor if available
    if _SAM2_CACHE["predictor"] is not None:
        predictor = _SAM2_CACHE["predictor"]
    else:
        # Initialize model
        checkpoint = "./sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        if not os.path.exists(checkpoint):
            print(f"[WARN] SAM2 checkpoint not found at {checkpoint}! Falling back to all-static mask.")
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

        # Add a union bounding box (person prompt) to capture the whole moving entity,
        # in addition to the individual object/part prompts.
        if len(boxes) > 1:
            boxes_np_tmp = np.array(boxes)
            ux1 = boxes_np_tmp[:, 0].min()
            uy1 = boxes_np_tmp[:, 1].min()
            ux2 = boxes_np_tmp[:, 2].max()
            uy2 = boxes_np_tmp[:, 3].max()
            boxes.append([ux1, uy1, ux2, uy2])

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


def compute_static_mask(rgb_paths, dataset_type="dex-ycb"):
    """
    Returns a boolean mask (H, W) where True = static across all frame transitions.
    """
    # For Hi4D, check if we can load pre-existing masks
    if dataset_type == "hi4d" and len(rgb_paths) > 0:
        first_path = rgb_paths[0]
        # Path structure: images/{cam_id}/{frame_id}.jpg
        # Mask structure: seg/img_seg_mask/{cam_id}/all/{frame_id}.png
        if "images" in first_path:
            parts = first_path.split(os.sep)
            try:
                # Find the index of "images" to resolve relative paths correctly
                img_idx = parts.index("images")
                subject_root = os.sep.join(parts[:img_idx])
                cam_id = parts[img_idx + 1]
                frame_id = os.path.splitext(parts[img_idx + 2])[0]

                mask_path = os.path.join(subject_root, "seg", "img_seg_mask", cam_id, "all", f"{frame_id}.png")
                if os.path.exists(mask_path):
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        # Hi4D masks: 255 for person, 0 for background
                        # We want True for static (background), False for dynamic (person)
                        return (m == 0)
            except (ValueError, IndexError):
                pass

    # Fallback to compute from frames
    frames = []
    for p in rgb_paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
    if len(frames) < 2:
        return None
    return compute_flow_sam2(frames)