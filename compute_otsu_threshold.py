import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu

# Step 1: Locate and import the GT validity mask function
from pi3.utils.gt import build_gt_validity_masks
from pi3.utils.camera_utils import discover_view_name

MODEL_NAME = "PI3"  # String constant for reuse


def compute_otsu():
    subject_name = "20200709-subject-01__20200709_141754"
    strategy = "baseline"
    views = 4

    # Import paths and threshold from the central config
    from eval_config import DATASET_BASE_ROOT, DEPTH_MAX_M

    # IMPORTANT: dataset_root must point to the ground truth data (RGB/Depth), NOT aligned_outputs
    dataset_root = os.path.join(DATASET_BASE_ROOT, subject_name)

    if not os.path.isdir(dataset_root):
        # Fallback to the hardcoded assets path just in case
        dataset_root = os.path.join("assets", "datasets", "dataset_pi3", subject_name)

    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Dataset root not found at {dataset_root}.")
        print("Please ensure this path contains the GT view folders (e.g., view_00/).")
        print(f"Current DATASET_BASE_ROOT is set to: {DATASET_BASE_ROOT}")
        return

    # Step 2: Load raw data
    # We'll try strategy-nested first, then flat
    possible_dirs = [
        os.path.join("aligned_outputs", strategy, subject_name, f"{views}views"),
        os.path.join("aligned_outputs", subject_name, f"{views}views"),
        os.path.join("aligned_outputs", strategy, subject_name),  # in case of different naming
    ]

    base_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            base_dir = d
            break

    if base_dir is None:
        print(f"[ERROR] Could not find reconstruction outputs in any of: {possible_dirs}")
        return

    paths = sorted(glob.glob(os.path.join(base_dir, "frame_*.npz")))
    if not paths:
        print(f"[ERROR] No .npz files found in {base_dir}")
        return

    print(f"Dataset root: {dataset_root}")
    print(f"Reconstruction dir: {base_dir}")
    print(f"Processing {len(paths)} frames...")

    all_valid_confs = []
    total_pixels_before = 0

    for path in paths:
        try:
            data = np.load(path)
            t = int(data['frame_idx'])

            # Raw per-pixel confidence, shape (V, H, W)
            confs = data['pointmaps_confs']
            V, H, W = confs.shape

            total_pixels_before += confs.size

            # Extract Ks to discover view names dynamically
            if 'Ks' in data:
                view_names = [discover_view_name(dataset_root, k) for k in data['Ks']]
                if any(v is None for v in view_names):
                    print(f"[WARN] Frame {t} contains unknown camera matrices (not found in dataset root). Skipping...")
                    continue
            else:
                print(f"[WARN] Frame {t} missing 'Ks' in .npz, skipping...")
                continue

            # Step 3: Apply GT validity mask
            # build_gt_validity_masks returns a list of boolean arrays or None
            vmasks = build_gt_validity_masks(t, view_names, dataset_root, depth_max_m=DEPTH_MAX_M, target_hw=(H, W))

            for v in range(V):
                if vmasks is not None and vmasks[v] is not None:
                    # Filter confidences using the GT mask
                    valid_conf = confs[v][vmasks[v]]
                    all_valid_confs.append(valid_conf)
                else:
                    # If no GT mask is available, we might skip or warn.
                    # For strict GT masking, we skip pixels without GT.
                    print(f"[WARN] Missing GT validity mask for frame {t}, view {view_names[v]}.")

        except Exception as e:
            print(f"[WARN] Failed to process {path}: {e}")
            continue

    if not all_valid_confs:
        print("[ERROR] No valid confidences extracted after GT masking.")
        return

    # Step 4: Pool confidences
    pooled_confs = np.concatenate(all_valid_confs)
    total_pixels_after = pooled_confs.size

    # Step 5: Compute Otsu threshold
    print("Computing Otsu threshold on pooled distribution...")
    otsu_thr = threshold_otsu(pooled_confs)

    kept_percentage = (np.sum(pooled_confs > otsu_thr) / total_pixels_after) * 100

    # Output stats
    print("\n--- Otsu Thresholding Results ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Total pixels before GT masking: {total_pixels_before:,}")
    print(f"Total pixels after GT masking:  {total_pixels_after:,}")
    print(f"Computed Otsu Threshold:        {otsu_thr:.6f}")
    print(f"Pixels kept at this threshold:  {kept_percentage:.2f}%\n")

    # Step 6: Visualise and report
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of the pooled confidence distribution with 200 bins
    counts, bins, patches = ax.hist(pooled_confs, bins=200, color='skyblue', edgecolor='none')

    # Vertical red dashed line at the Otsu threshold
    ax.axvline(otsu_thr, color='red', linestyle='--', linewidth=2, label=f'Otsu Thr = {otsu_thr:.4f}')

    ax.set_title(f'Confidence Distribution & Otsu Threshold ({MODEL_NAME})\nThreshold: {otsu_thr:.6f}')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Pixel count')
    ax.legend()

    out_img = 'otsu_threshold_analysis.png'
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {out_img}")
    # plt.show() # Optional


if __name__ == "__main__":
    compute_otsu()
