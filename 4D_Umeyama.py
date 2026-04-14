import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import rerun as rr

# Path setup for MASt3R/DUSt3R
import mast3r.utils.path_to_dust3r  # noqa

from mast3r.utils.umeyama_alignment import apply_similarity_transform
from mast3r.utils.gt import load_gt_params, build_gt_validity_masks
from mast3r.utils.camera_utils import discover_view_name
from mast3r.utils.alignment_4d import (
    strategy1_reference,
    strategy2_hierarchical,
    solve_final_gt_registration,
    compute_4d_jitter_complete,
    normalize_spatial_dims,
    normalize_array
)
from mast3r.utils.temporal_metrics import (
    compute_chamfer_distance,
    compute_accuracy,
    compute_completeness,
    split_points_by_mask
)
from eval_config import DATASET_BASE_ROOT, SUBJECT_NAMES, SUBJECT_BY_CODE, MIN_CONF_THR


# ── camera discovery ───────────────────────────────────────────────────────────
# Moved to utils.camera_utils

# ── evaluation metrics ────────────────────────────────────────────────────────
# Now using utils.temporal_metrics

# ── correspondence extraction ─────────────────────────────────────────────────
# Moved to utils.alignment_4d

# ── alignment strategies ──────────────────────────────────────────────────────
# Moved to utils.alignment_4d

def evaluate_4d_sequence(frame_npz_paths, frame_transforms, s_glob, R_glob, tr_glob, dataset_root):
    """
    Evaluates the final 4D sequence against ground truth.
    """
    chamfer_list, static_acc_list, completeness_list = [], [], []

    for i, path in enumerate(frame_npz_paths):
        data = np.load(path)
        V, H, W = normalize_spatial_dims(data)
        if H == 0: continue

        # 1. Aligned Estimated Points
        pm = normalize_array(data['pointmaps'], V, H, W).astype(np.float32)
        s_i, R_i, tr_i = frame_transforms[i]
        s_total, R_total, tr_total = s_glob * s_i, R_glob @ R_i, s_glob * (R_glob @ tr_i) + tr_glob

        pts_final = apply_similarity_transform(pm.reshape(-1, 3), s_total, R_total, tr_total)

        # 2. GT Points (Filtered by validity inside metrics helper)
        gt_pts = data['gt_pts']
        if np.mean(np.abs(gt_pts)) > 10.0: gt_pts /= 1000.0

        t, ks, rts = int(data['frame_idx']), data['Ks'], data['R_ts']
        view_names = [discover_view_name(dataset_root, k) for k in ks]
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, target_hw=(H, W))

        # Stricter static+valid mask for evaluation
        m_eval = normalize_array(data['masks_2d'], V, H, W, is_mask=True)
        for v in range(V):
            if vmasks[v] is not None: m_eval[v] &= vmasks[v]
            if 'pointmaps_confs' in data:
                conf = normalize_array(data['pointmaps_confs'], V, H, W)
                m_eval[v] &= (conf[v] > MIN_CONF_THR)

        # Split and calculate
        s_p, _ = split_points_by_mask(pts_final, m_eval, ks, rts)
        g_s, _ = split_points_by_mask(gt_pts, m_eval, ks, rts)

        if len(g_s) > 0 and len(s_p) > 0:
            chamfer_list.append(compute_chamfer_distance(s_p, g_s))
            static_acc_list.append(compute_accuracy(s_p, g_s))
            completeness_list.append(compute_completeness(s_p, g_s))
        else:
            print(f"  [WARN] Frame {i}: insufficient static points (est={len(s_p)}, gt={len(g_s)})")

    return {
        'chamfer': np.nanmean(chamfer_list),
        'static_acc': np.nanmean(static_acc_list),
        'completeness': np.nanmean(completeness_list),
    }


# ── visualization ─────────────────────────────────────────────────────────────

def log_gt_sequence(paths):
    entity = "4d_eval/GT"
    for p in paths:
        data = np.load(p)
        t = int(data['frame_idx'])
        rr.set_time("timestep", sequence=t)
        gt_pts = data['gt_pts']
        # Magnitude check: if any point's distance from origin is > 10m, it's millimeters
        if np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
            gt_pts = gt_pts / 1000.0
        rr.log(entity, rr.Points3D(positions=gt_pts, colors=[0, 255, 0], radii=0.002))


def log_aligned_sequence(paths, frame_transforms, s_glob, R_glob, tr_glob, label, color, dataset_root):
    entity = f"4d_eval/{label}"
    for i, p in enumerate(paths):
        data = np.load(p)
        V, H, W = normalize_spatial_dims(data)
        if H == 0: continue

        pm = normalize_array(data['pointmaps'], V, H, W).astype(np.float32)
        t, ks = int(data['frame_idx']), data['Ks']
        rr.set_time("timestep", sequence=t)

        view_names = [discover_view_name(dataset_root, k) for k in ks]
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, target_hw=(H, W))

        s_i, R_i, tr_i = frame_transforms[i]
        s_total = s_glob * s_i
        R_total = R_glob @ R_i
        tr_total = s_glob * (R_glob @ tr_i) + tr_glob

        m_base = normalize_array(data['masks_2d'], V, H, W, is_mask=True)
        if 'pointmaps_confs' in data:
            m_base &= (normalize_array(data['pointmaps_confs'], V, H, W) > MIN_CONF_THR)

        for v in range(V):
            mask = m_base[v]
            if vmasks[v] is not None:
                mask &= vmasks[v]

            pts = pm[v][mask]
            if len(pts) == 0: continue
            pts_final = apply_similarity_transform(pts, s_total, R_total, tr_total)
            rr.log(f"{entity}/view_{v}", rr.Points3D(positions=pts_final, colors=color, radii=0.002))


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate_subject(subject_name):
    in_dir = os.path.join("aligned_outputs", subject_name, "2views")
    if not os.path.exists(in_dir): return
    paths = sorted(glob.glob(os.path.join(in_dir, "frame_*.npz")),
                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    if len(paths) < 2: return

    dataset_root = os.path.join(DATASET_BASE_ROOT, subject_name)
    rr.init(f"MASt3R_4D_{subject_name}", spawn=True)

    # 0. GT
    print("  [RERUN] Logging GT...")
    log_gt_sequence(paths)

    results = []

    # 1. Strategy 1 (Reference Frame 0)
    print("\n--- [Strategy 1] Reference Frame Alignment ---")
    t1_start = time.time()
    tf_s1 = strategy1_reference(paths, dataset_root)
    s_g1, R_g1, tr_g1 = solve_final_gt_registration(paths, tf_s1, dataset_root)

    metrics_s1 = evaluate_4d_sequence(paths, tf_s1, s_g1, R_g1, tr_g1, dataset_root)
    jitter_s1 = compute_4d_jitter_complete(paths, tf_s1, s_g1, R_g1, tr_g1, dataset_root)
    t1_end = time.time()

    print(f"  [4D-S1] Chamfer: {metrics_s1['chamfer']:.5f}")
    print(f"  [4D-S1] Static Acc: {metrics_s1['static_acc']:.4f}")
    print(f"  [4D-S1] Jitter Mean: {jitter_s1['jitter_mean']:.5f}")

    log_aligned_sequence(paths, tf_s1, s_g1, R_g1, tr_g1, "Strategy_1", [0, 0, 255], dataset_root)

    results.append({
        'subject': subject_name, 'strategy': 'S1_Global',
        'scale': s_g1, 'solve_time': t1_end - t1_start,
        'chamfer': metrics_s1['chamfer'], 'static_acc': metrics_s1['static_acc'],
        'jitter_mean': jitter_s1['jitter_mean']
    })

    # 2. Strategy 2 (Hierarchical)
    print("\n--- [Strategy 2] Hierarchical Alignment ---")
    t2_start = time.time()
    tf_s2 = strategy2_hierarchical(paths, dataset_root)
    s_g2, R_g2, tr_g2 = solve_final_gt_registration(paths, tf_s2, dataset_root)

    metrics_s2 = evaluate_4d_sequence(paths, tf_s2, s_g2, R_g2, tr_g2, dataset_root)
    jitter_s2 = compute_4d_jitter_complete(paths, tf_s2, s_g2, R_g2, tr_g2, dataset_root)
    t2_end = time.time()

    print(f"  [4D-S2] Chamfer: {metrics_s2['chamfer']:.5f}")
    print(f"  [4D-S2] Static Acc: {metrics_s2['static_acc']:.4f}")
    print(f"  [4D-S2] Jitter Mean: {jitter_s2['jitter_mean']:.5f}")

    log_aligned_sequence(paths, tf_s2, s_g2, R_g2, tr_g2, "Strategy_2", [255, 0, 255], dataset_root)

    results.append({
        'subject': subject_name, 'strategy': 'S2_Hierarchical',
        'scale': s_g2, 'solve_time': t2_end - t2_start,
        'chamfer': metrics_s2['chamfer'], 'static_acc': metrics_s2['static_acc'],
        'jitter_mean': jitter_s2['jitter_mean']
    })

    # Save results to CSV
    df = pd.DataFrame(results)
    out_csv = f"eval_4d_{subject_name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved results to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    for code in SUBJECT_BY_CODE.keys(): parser.add_argument(f"--{code}", action="store_true")
    args = parser.parse_args()

    selected = [v for k, v in SUBJECT_BY_CODE.items() if getattr(args, k)]
    if args.all: selected = SUBJECT_NAMES
    for s in selected: evaluate_subject(s)
