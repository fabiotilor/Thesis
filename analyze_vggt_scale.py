import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Configuration
SUBJECT_NAME = "20201002-subject-08__20201002_110227"
NUM_VIEWS = 2
ALIGNED_ROOT = "aligned_outputs/baseline"
MODEL_RES = 518


def analyze_vggt():
    subject_dir = os.path.join(ALIGNED_ROOT, SUBJECT_NAME, f"{NUM_VIEWS}views")
    # Handle different possible frame naming conventions (frame_00.npz or frame_0000.npz)
    frame_files = sorted(glob.glob(os.path.join(subject_dir, "frame_*.npz")))

    if not frame_files:
        print(f"No frames found in {subject_dir}")
        return

    f_est_list, f_gt_list = [], []
    eigenvalues_list = []
    frames = []

    print(f"Analyzing {len(frame_files)} frames for {SUBJECT_NAME}...")

    for fpath in frame_files:
        try:
            data = np.load(fpath)
            t = int(data['frame_idx'])

            # 1. Focal Length Analysis
            # est_intrinsics shape (V, 3, 3)
            if 'est_intrinsics' not in data or 'Ks' not in data:
                continue

            est_ks = data['est_intrinsics']
            gt_ks = data['Ks']

            # Scale factor: Model is 518x518, Dex-YCB sensor is 640x480
            sensor_w = 640
            scale_factor = MODEL_RES / sensor_w

            # We take the mean across available views for this frame
            f_est = np.mean(est_ks[:, 0, 0])
            f_gt_scaled = np.mean(gt_ks[:, 0, 0]) * scale_factor

            f_est_list.append(f_est)
            f_gt_list.append(f_gt_scaled)
            frames.append(t)

            # 2. Planarity Analysis (Covariance Eigenvalues)
            if 'aligned_pts' in data:
                pts = data['aligned_pts']
                if len(pts) > 10:
                    # Center the points
                    pts_centered = pts - np.mean(pts, axis=0)
                    # Covariance matrix
                    cov = (pts_centered.T @ pts_centered) / len(pts)
                    # Eigenvalues (sorted ascending)
                    evals = np.sort(np.linalg.eigvals(cov))
                    eigenvalues_list.append(evals)
                else:
                    eigenvalues_list.append([0, 0, 0])
            else:
                eigenvalues_list.append([0, 0, 0])
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    if not f_est_list:
        print("No valid data extracted.")
        return

    # --- Plotting ---
    plt.figure(figsize=(15, 6))

    # Plot 1: Focal Lengths
    plt.subplot(1, 2, 1)
    plt.plot(frames, f_est_list, label='VGGT Predicted Focal', color='red', alpha=0.7)
    plt.plot(frames, f_gt_list, label='GT Focal (Scaled to 518px)', color='green', linestyle='--', linewidth=2)
    plt.title(f"Focal Length Comparison\n{SUBJECT_NAME} ({NUM_VIEWS} views)")
    plt.xlabel("Frame index")
    plt.ylabel("Focal length (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Smallest Eigenvalue (Planarity)
    plt.subplot(1, 2, 2)
    small_evals = [e[0] for e in eigenvalues_list]
    plt.plot(frames, small_evals, color='blue', label='Lambda_min (Smallest Variance)')
    plt.axhline(y=0.0001, color='orange', linestyle=':', label='Near-Planar (0.0001)')
    plt.title("Scene Planarity Analysis\n(Small eigenvalue = Ill-posed scale)")
    plt.xlabel("Frame index")
    plt.ylabel("Eigenvalue Magnitude")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    plt.tight_layout()
    out_plot = f"vggt_diagnostics_{SUBJECT_NAME.split('subject-')[1][:2]}.png"
    plt.savefig(out_plot)
    print(f"\nAnalysis plot saved to: {out_plot}")

    # Summary Statistics
    f_err_pct = np.mean(np.abs(np.array(f_est_list) - np.array(f_gt_list)) / np.array(f_gt_list)) * 100
    avg_min_ev = np.mean(small_evals)

    print(f"\n--- Diagnostic Summary ---")
    print(f"Average Focal Error: {f_err_pct:.2f}%")
    print(f"Average Lambda_min:  {avg_min_ev:.8f}")

    if f_err_pct > 5:
        print("CRITICAL: Focal length prediction is significantly biased. This is the primary cause of scale issues.")
    if avg_min_ev < 0.0005:
        print("WARNING: Scene is nearly planar. Umeyama alignment is numerically unstable (ill-posed).")
    else:
        print("INFO: Scene has sufficient 3D structure for stable alignment.")


if __name__ == "__main__":
    analyze_vggt()
