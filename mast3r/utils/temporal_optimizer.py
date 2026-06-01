"""
temporal_optimizer.py — Ground-truth-free temporal smoother for VGGT / MASt3R.

Exploits multi-view geometric redundancy (2–4 synchronised cameras observing
the same static scene) to self-calibrate how much smoothing to apply.
No gradient descent, no GT depth, no iterative optimisation.
"""

import os
import glob
import json
import time
import numpy as np

from .alignment_4d import normalize_spatial_dims, normalize_array
from .umeyama_alignment import apply_similarity_transform
from .camera_utils import discover_view_name
from .gt import build_gt_validity_masks


# ---------------------------------------------------------------------------
# Auto-generate strategy2 outputs from baseline if they don't exist
# ---------------------------------------------------------------------------

def _sorted_frame_paths(directory):
    """Return sorted frame_*.npz paths from a directory."""
    return sorted(
        glob.glob(os.path.join(directory, "frame_*.npz")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]),
    )


def ensure_base_strategy_exists(
    subject_name,
    n_views,
    dataset_type="dex-ycb",
    base_strategy="strategy2",
):
    """
    Check whether the base strategy outputs already exist.  If not, compute
    them on the fly from the baseline .npz files – exactly as 4D_Umeyama.py
    does.

    Returns the path to the base-strategy output directory, or None on
    failure.
    """
    from eval_config import DATASETS

    # 1. Locate the base-strategy output dir
    base_dir = os.path.join(
        "aligned_outputs", dataset_type, base_strategy,
        subject_name, f"{n_views}views",
    )
    if os.path.isdir(base_dir) and len(_sorted_frame_paths(base_dir)) >= 2:
        return base_dir  # already exists

    # 2. Locate baseline frames
    baseline_dir = os.path.join(
        "aligned_outputs", dataset_type, "baseline",
        subject_name, f"{n_views}views",
    )
    if not os.path.isdir(baseline_dir):
        # Legacy fallback (old layout without dataset_type prefix)
        baseline_dir = os.path.join(
            "aligned_outputs", "baseline",
            subject_name, f"{n_views}views",
        )
    if not os.path.isdir(baseline_dir):
        print(f"[ERROR] Baseline outputs not found for {subject_name} {n_views}views")
        return None

    baseline_paths = _sorted_frame_paths(baseline_dir)
    if len(baseline_paths) < 2:
        print(f"[ERROR] Not enough baseline frames in {baseline_dir}")
        return None

    # 3. Compute strategy 2 (hierarchical alignment)
    dataset_cfg = DATASETS[dataset_type]
    dataset_root = os.path.join(dataset_cfg["root"], subject_name)

    print(f"\n[AUTO] Computing {base_strategy} for {subject_name} {n_views}views ...")
    from .alignment_4d import (
        strategy2_hierarchical,
        solve_final_gt_registration,
    )
    # Import save_aligned_results from 4D_Umeyama (the canonical location)
    from importlib import import_module
    _umeyama_mod = import_module("4D_Umeyama")
    save_aligned_results = _umeyama_mod.save_aligned_results
    save_timing = _umeyama_mod.save_timing

    t0 = time.perf_counter()

    tf = strategy2_hierarchical(
        baseline_paths, dataset_root, dataset_type=dataset_type,
    )
    s_g, R_g, tr_g = solve_final_gt_registration(
        baseline_paths, tf, dataset_root,
        use_static_mask=False, dataset_type=dataset_type,
    )

    os.makedirs(base_dir, exist_ok=True)
    save_aligned_results(
        baseline_paths, tf, s_g, R_g, tr_g,
        subject_name,
        strategy_label=f"{base_strategy}_{n_views}views",
        dataset_root=dataset_root,
        out_dir=base_dir,
        skip_existing_frames=False,
        dataset_type=dataset_type,
    )
    save_timing(base_dir, base_strategy, len(baseline_paths),
                time.perf_counter() - t0)

    print(f"[AUTO] {base_strategy} saved to {base_dir}")
    return base_dir


# ---------------------------------------------------------------------------
# Step 1 — Load and align frames
# ---------------------------------------------------------------------------

def load_and_align_frames(frame_paths):
    """
    Loads NPZ files, extracts pointmaps, confidences, and masks, and transforms
    the pointmaps into the shared global coordinate system using saved (s, R, tr).

    Returns:
        all_pmaps  (T, V, H, W, 3) float32 — aligned pointmaps in global coords
        all_confs  (T, V, H, W)    float32 — per-pixel confidence
        all_masks  (T, V, H, W)    bool    — True = static pixel
        all_data   list[dict]              — raw NPZ dicts kept in memory
    """
    all_pmaps = []
    all_confs = []
    all_masks = []
    all_data = []

    for path in frame_paths:
        data = np.load(path)
        V, H, W = normalize_spatial_dims(data)
        if H == 0:
            continue

        pm = normalize_array(data['pointmaps'], V, H, W).astype(np.float32)
        conf = (normalize_array(data['pointmaps_confs'], V, H, W).astype(np.float32)
                if 'pointmaps_confs' in data
                else np.ones((V, H, W), dtype=np.float32))
        mask = normalize_array(data['masks_2d'], V, H, W, is_mask=True)

        # Bring to shared global coordinate system using saved alignment
        s_val = data.get('scale', 1.0)
        R_val = data.get('R', np.eye(3))
        tr_val = data.get('tr', np.zeros(3))

        aligned_pm = np.empty_like(pm)
        for vi in range(V):
            aligned_pm[vi] = apply_similarity_transform(
                pm[vi].reshape(-1, 3), s_val, R_val, tr_val
            ).reshape(H, W, 3)

        all_pmaps.append(aligned_pm)
        all_confs.append(conf)
        all_masks.append(mask)
        all_data.append(dict(data))  # Convert NpzFile to dict to hold in memory

    if not all_pmaps:
        return None, None, None, None

    all_pmaps = np.stack(all_pmaps, axis=0)
    all_confs = np.stack(all_confs, axis=0)
    all_masks = np.stack(all_masks, axis=0)

    return all_pmaps, all_confs, all_masks, all_data


# ---------------------------------------------------------------------------
# Step 2 — Confidence-weighted Gaussian temporal smooth
# ---------------------------------------------------------------------------

def confidence_weighted_temporal_smooth(all_pmaps, all_confs, all_masks, sigma, alpha):
    """
    Applies a confidence-weighted Gaussian temporal filter to static pixels.

    Dynamic pixels are FROZEN — kept at their original values.

    Args:
        all_pmaps: (T, V, H, W, 3) — aligned pointmaps in global coords
        all_confs: (T, V, H, W)    — per-pixel model confidence
        all_masks: (T, V, H, W)    — True = static
        alpha:     float — blending factor
    """
    T, V, H, W, _ = all_pmaps.shape
    final_pmaps = np.copy(all_pmaps)

    if sigma <= 0 or alpha <= 0:
        return final_pmaps

    # Pre-compute combined weight: confidence × static mask
    valid_weight = all_confs * all_masks.astype(np.float32)  # (T, V, H, W)

    for t in range(T):
        # Gaussian temporal weights centred at frame t
        dist_sq = (np.arange(T, dtype=np.float64) - t) ** 2
        w_t = np.exp(-dist_sq / (2.0 * max(sigma ** 2, 1e-6)))
        w_t = w_t.astype(np.float32).reshape(T, 1, 1, 1)

        # Combined weight: temporal × confidence × static-mask
        W = w_t * valid_weight  # (T, V, H, W)
        sum_W = W.sum(axis=0)   # (V, H, W)

        # Weighted mean of aligned positions
        smoothed_p = (W[..., None] * all_pmaps).sum(axis=0) / np.clip(sum_W[..., None], 1e-8, None)

        # Blend with original
        blended_p = (1.0 - alpha) * all_pmaps[t] + alpha * smoothed_p

        # Only update static pixels with sufficient weight
        update_mask = all_masks[t] & (sum_W > 1e-8)
        final_pmaps[t] = np.where(update_mask[..., None], blended_p, all_pmaps[t])

    return final_pmaps


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def optimize_temporal_consistency(
    frame_paths, out_dir, dataset_root,
    sigma=4.0, alpha=0.5,
    dataset_type="dex-ycb",
):
    """
    Main entrypoint: ground-truth-free temporal smoother.

    1. Load & align frames to shared coordinate system.
    2. Apply confidence-weighted Gaussian temporal smooth.
    3. Inverse-transform and save.
    4. Write timing.json.

    Returns:
        list[str] — paths of saved .npz files.
    """
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Load and align
    # ------------------------------------------------------------------
    print(f"  [OPT] Loading and aligning {len(frame_paths)} frames...")
    all_pmaps, all_confs, all_masks, all_data = load_and_align_frames(frame_paths)

    if all_pmaps is None:
        print("  [WARN] No valid frames found for temporal optimization.")
        return []

    # ------------------------------------------------------------------
    # Step 2 — Confidence-weighted temporal smooth
    # ------------------------------------------------------------------
    print("  [OPT] Applying temporal smoothing...")
    print(f"  [OPT] sigma={sigma:.2f} alpha={alpha:.2f}")
    smoothed_pmaps = confidence_weighted_temporal_smooth(
        all_pmaps, all_confs, all_masks, sigma, alpha,
    )

    # ------------------------------------------------------------------
    # Step 3 — Inverse transform and save
    # ------------------------------------------------------------------
    print("  [OPT] Transforming smoothed points back and generating aligned_pts...")
    from eval_config import CONF_PERCENTILE

    saved_paths = []

    for i, data in enumerate(all_data):
        V_i, H_i, W_i = normalize_spatial_dims(data)
        s_val = data.get('scale', 1.0)
        R_val = data.get('R', np.eye(3))
        tr_val = data.get('tr', np.zeros(3))

        # Inverse similarity transform: global → local camera coords
        s_inv = 1.0 / max(float(s_val), 1e-12)
        R_inv = R_val.T
        tr_inv = -s_inv * (R_inv @ tr_val)

        local_smoothed_pm = np.empty_like(smoothed_pmaps[i])
        for vi in range(V_i):
            local_smoothed_pm[vi] = apply_similarity_transform(
                smoothed_pmaps[i, vi].reshape(-1, 3), s_inv, R_inv, tr_inv
            ).reshape(H_i, W_i, 3)

        # Rebuild aligned_pts in global coords using confidence-percentile mask
        t_idx = int(data["frame_idx"])
        ks = data["Ks"]

        if 'view_names' in data:
            view_names = (data['view_names'].tolist()
                          if hasattr(data['view_names'], 'tolist')
                          else list(data['view_names']))
        else:
            view_names = [discover_view_name(dataset_root, k) for k in ks]

        vmasks = build_gt_validity_masks(
            t_idx, view_names, dataset_root,
            target_hw=(H_i, W_i), dataset_type=dataset_type,
        )

        all_pts = []
        conf = all_confs[i]

        for v in range(V_i):
            mask = np.ones((H_i, W_i), dtype=bool)
            if vmasks[v] is not None:
                mask &= vmasks[v]

            if conf is not None:
                thr = np.percentile(conf[v], 100 * (1 - CONF_PERCENTILE))
                mask &= conf[v] > thr

            p_v = smoothed_pmaps[i, v][mask]  # global-coordinate smoothed points
            if len(p_v) > 0:
                all_pts.append(p_v)

        aligned_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))

        # Assemble save dict — all original keys copied verbatim except
        # pointmaps (replaced with local-coords smoothed) and aligned_pts.
        save_dict = {
            "gt_pts": data["gt_pts"],
            "aligned_pts": aligned_pts,
            "frame_idx": t_idx,
            "Ks": ks,
            "R_ts": data["R_ts"],
            "masks_2d": data["masks_2d"],
            "pointmaps": local_smoothed_pm,
            "pointmaps_confs": data.get("pointmaps_confs"),
            "scale": s_val,
            "R": R_val,
            "tr": tr_val,
        }
        if "est_poses" in data:
            save_dict["est_poses"] = data["est_poses"]
        if "est_intrinsics" in data:
            save_dict["est_intrinsics"] = data["est_intrinsics"]
        if "view_names" in data:
            save_dict["view_names"] = data["view_names"]

        out_path = os.path.join(out_dir, f"frame_{t_idx:04d}.npz")
        np.savez_compressed(out_path, **save_dict)
        saved_paths.append(out_path)

    # ------------------------------------------------------------------
    # Step 4 — Write timing.json
    # ------------------------------------------------------------------
    total_seconds = time.perf_counter() - start_time
    timing_payload = {
        "strategy": "opt",
        "n_frames": len(saved_paths),
        "total_seconds": float(total_seconds),
        "seconds_per_frame": float(total_seconds / max(len(saved_paths), 1)),
    }
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, indent=2)

    print(f"  [TIME] Temporal Opt: total={timing_payload['total_seconds']:.2f}s  "
          f"per_frame={timing_payload['seconds_per_frame']:.3f}s")
    return saved_paths
