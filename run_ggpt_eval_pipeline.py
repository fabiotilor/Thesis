#!/usr/bin/env python3
"""
GGPT Refinement + Alignment + Evaluation Pipeline.

Usage:
    python run_ggpt_eval_pipeline.py --subject all --views 2 3 4 --model vggt
    python run_ggpt_eval_pipeline.py --subject 01 --views 2 --model mast3r
    python run_ggpt_eval_pipeline.py --subject 01 --views 2 --model pi3
    python run_ggpt_eval_pipeline.py --subject 01 --views 2 --model pi3x
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "scripts")
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eval_config import (
    SUBJECT_BY_CODE, SUBJECT_NAMES, DATASET_BASE_ROOT,
    RERUN_ADDR, RERUN_EYE_UP,
    get_dataset_config, get_subject_by_code, get_dataset_root_for_subject,
    get_view_config, get_pair_name_for_subject,
    CONF_PERCENTILE,
)
from utils.umeyama_alignment import estimate_similarity_transform, apply_similarity_transform
from utils.alignment_4D import (
    strategy1_reference, strategy2_hierarchical, strategy3_pgo,
    solve_final_gt_registration, normalize_spatial_dims, normalize_array,
    extract_clean_gt_correspondences, get_view_names_and_masks
)
from utils.camera_utils import discover_view_name
from utils.gt import build_gt_validity_masks
from utils.temporal_metrics import (
    compute_chamfer_distance, compute_accuracy, compute_completeness,
    split_points_by_mask, compute_static_jitter, compute_camera_metrics
)

try:
    import rerun as rr
    from utils.rerun_logging import (
        init_recording, configure_rerun_view_defaults,
        log_pointcloud, log_gt_sequence, log_aligned_sequence
    )
except ImportError:
    rr = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sorted_frame_paths(frame_dir: str):
    paths = glob.glob(os.path.join(frame_dir, "frame_*.npz"))
    if not paths:
        return []
    return sorted(paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]))


def _write_timing_json(out_dir, label, n_frames, total_seconds):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "strategy": label,
        "n_frames": int(n_frames),
        "total_seconds": float(total_seconds),
        "seconds_per_frame": float(total_seconds / max(n_frames, 1)),
    }
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(payload, f, indent=2)


# ── Per-frame baseline alignment (Umeyama to GT) ─────────────────────────────

def run_baseline_alignment(frame_paths, dataset_root, out_dir, skip_existing=True, dataset_type="dex-ycb"):
    """
    Per-frame Umeyama alignment of GGPT-refined pointmaps to GT.
    Unlike align_reconstruction_umeyama.py, confidence filtering and validity
    masking are NOT re-applied here because the refined outputs already went
    through that filtering in prepare_ggpt_inputs.py.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"  [BASELINE] Aligning {len(frame_paths)} frames -> {out_dir}")

    for path in frame_paths:
        data = np.load(path)
        t = int(data["frame_idx"])
        out_path = os.path.join(out_dir, f"frame_{t:05d}.npz")

        if skip_existing and os.path.exists(out_path):
            continue

        V, H, W = normalize_spatial_dims(data)
        if H == 0:
            continue

        pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
        conf = normalize_array(data["pointmaps_confs"], V, H, W) if "pointmaps_confs" in data else None
        m_2d = normalize_array(data["masks_2d"], V, H, W, is_mask=True)
        ks = data["Ks"]

        view_names, vmasks = get_view_names_and_masks(data, dataset_root, dataset_type=dataset_type)

        # In DexYCB, None masks mean no depth map (invalid). In self-contained
        # datasets, None masks mean there is no external mask and all pixels may be used.
        if dataset_type == "dex-ycb" and all(m is None for m in vmasks):
            print(f"    [WARN] Frame {t}: Could not resolve any view names/masks, skipping.")
            continue

        # Build correspondences for Umeyama (using already-filtered points)
        res = extract_clean_gt_correspondences(data, dataset_root, use_static_mask=False, dataset_type=dataset_type)
        if res is None:
            # print(f"    [WARN] Frame {t}: No correspondences found, skipping.")
            continue
        src, dst = res
        if len(src) < 6:
            continue

        s, R, tr = estimate_similarity_transform(src, dst)

        # Build aligned point cloud
        all_pts = []
        for v in range(V):
            mask = np.ones((H, W), dtype=bool)
            if vmasks[v] is not None:
                mask &= vmasks[v]
            p_v = pm[v][mask]
            if len(p_v) > 0:
                all_pts.append(apply_similarity_transform(p_v, s, R, tr))

        aligned_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))

        # Build GT pointcloud for this frame
        if dataset_type == "hi4d":
            from models.scripts.utils.gt import _load_hi4d_mesh_gt
            gt_pts = _load_hi4d_mesh_gt(dataset_root, t)
            if gt_pts is None:
                gt_pts = np.zeros((0, 3))
        else:
            gt_pts = data["gt_pts"]
            if gt_pts.ndim == 4:
                # (V, H, W, 3) -> flatten valid
                gt_mask_all = np.linalg.norm(gt_pts, axis=-1) > 1e-6
                for v in range(V):
                    if vmasks[v] is not None:
                        gt_mask_all[v] &= vmasks[v]
                gt_pts = gt_pts[gt_mask_all]
            # Only convert mm->m for dex-ycb.
            if dataset_type == "dex-ycb" and np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
                gt_pts = gt_pts / 1000.0

        save_dict = {
            "gt_pts": gt_pts,
            "aligned_pts": aligned_pts,
            "scale": float(s),
            "R": R,
            "tr": tr,
            "pointmaps": data["pointmaps"],
            "pointmaps_confs": data.get("pointmaps_confs"),
            "frame_idx": t,
            "Ks": ks,
            "R_ts": data["R_ts"],
            "masks_2d": data["masks_2d"],
            "est_poses": data.get("est_poses"),
            "est_intrinsics": data.get("est_intrinsics"),
        }
        np.savez(out_path, **save_dict)

    return _sorted_frame_paths(out_dir)


# ── Strategy alignment + save ─────────────────────────────────────────────────

def run_strategy_alignment(baseline_paths, dataset_root, strategy_func, out_dir,
                           strategy_label, skip_existing=True, dataset_type="dex-ycb", **kwargs):
    """
    Run a 4D alignment strategy, then save per-frame .npz with aligned_pts.
    Uses save logic from 4D_Umeyama.py's save_aligned_results.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Check if all frames already exist
    if skip_existing:
        existing = _sorted_frame_paths(out_dir)
        if len(existing) >= len(baseline_paths):
            print(f"  [SKIP] {strategy_label}: {len(existing)} frames already exist in {out_dir}")
            return existing

    t0 = time.perf_counter()
    frame_transforms = strategy_func(baseline_paths, dataset_root, dataset_type=dataset_type, **kwargs)
    s_glob, R_glob, tr_glob = solve_final_gt_registration(
        baseline_paths, frame_transforms, dataset_root, use_static_mask=False,
        dataset_type=dataset_type
    )

    for i, path in enumerate(baseline_paths):
        data = np.load(path)
        V, H, W = normalize_spatial_dims(data)
        if H == 0:
            continue
        t = int(data["frame_idx"])

        pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
        conf = normalize_array(data["pointmaps_confs"], V, H, W) if "pointmaps_confs" in data else None
        ks = data["Ks"]

        # Robustly get view names and masks
        view_names, vmasks = get_view_names_and_masks(data, dataset_root, dataset_type=dataset_type)

        s_i, R_i, tr_i = frame_transforms[i]
        s_tot = s_glob * s_i
        R_tot = R_glob @ R_i
        tr_tot = s_glob * (R_glob @ tr_i) + tr_glob

        all_pts = []
        for v in range(V):
            mask = np.ones((H, W), dtype=bool)
            if vmasks[v] is not None:
                mask &= vmasks[v]
            p_v = pm[v][mask]
            if len(p_v) > 0:
                all_pts.append(apply_similarity_transform(p_v, s_tot, R_tot, tr_tot))

        aligned_pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))

        gt_pts = data["gt_pts"]
        if dataset_type == "dex-ycb" and np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
            gt_pts = gt_pts / 1000.0

        save_dict = {
            "gt_pts": gt_pts,
            "aligned_pts": aligned_pts,
            "scale": float(s_tot),
            "R": R_tot,
            "tr": tr_tot,
            "pointmaps": data["pointmaps"],
            "pointmaps_confs": data.get("pointmaps_confs"),
            "frame_idx": t,
            "Ks": ks,
            "R_ts": data["R_ts"],
            "masks_2d": data["masks_2d"],
            "est_poses": data.get("est_poses"),
            "est_intrinsics": data.get("est_intrinsics"),
        }
        out_path = os.path.join(out_dir, f"frame_{t:05d}.npz")
        np.savez(out_path, **save_dict)

    elapsed = time.perf_counter() - t0
    _write_timing_json(out_dir, strategy_label, len(baseline_paths), elapsed)
    return _sorted_frame_paths(out_dir)


# ── Evaluation (adapted from evaluate_4D.py) ─────────────────────────────────

def _stringify_array_metric(metrics, key):
    if key in metrics:
        metrics[key] = ",".join(str(int(v)) for v in np.asarray(metrics[key]).ravel())


def _jitter_inputs_from_dir(in_dir, dataset_root, dataset_type="dex-ycb"):
    files = _sorted_frame_paths(in_dir)
    all_pm_mv, all_masks_mv, all_validity_masks_mv, all_confs_mv = [], [], [], []

    for f in files:
        data = np.load(f)
        if "pointmaps" not in data or "masks_2d" not in data:
            continue

        V, H, W = normalize_spatial_dims(data)
        if H == 0:
            continue

        pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
        conf = normalize_array(data["pointmaps_confs"], V, H, W) if "pointmaps_confs" in data else None

        m_2d_raw = data["masks_2d"]
        m_2d_static = ~m_2d_raw if dataset_type == "hi4d" else m_2d_raw
        masks = normalize_array(m_2d_static, V, H, W, is_mask=True)

        s_val = data["scale"] if "scale" in data else 1.0
        R_val = data["R"] if "R" in data else np.eye(3)
        tr_val = data["tr"] if "tr" in data else np.zeros(3)

        aligned_pm = np.empty_like(pm)
        for vi in range(V):
            aligned_pm[vi] = apply_similarity_transform(
                pm[vi].reshape(-1, 3), s_val, R_val, tr_val
            ).reshape(H, W, 3)

        all_pm_mv.append(aligned_pm)
        all_masks_mv.append(masks.astype(bool))
        if conf is not None:
            all_confs_mv.append(conf)

        _, vmasks = get_view_names_and_masks(data, dataset_root, dataset_type=dataset_type)
        all_validity_masks_mv.append(np.array([
            vmask if vmask is not None else np.ones((H, W), dtype=bool)
            for vmask in vmasks
        ], dtype=bool))

    return files, all_pm_mv, all_masks_mv, all_validity_masks_mv, all_confs_mv


def evaluate_jitter_strategy_dir(in_dir, strategy_label, dataset_root, dataset_type="dex-ycb"):
    files, all_pm_mv, all_masks_mv, all_validity_masks_mv, all_confs_mv = _jitter_inputs_from_dir(
        in_dir, dataset_root, dataset_type=dataset_type,
    )
    if not files:
        return None

    print(f"    [JITTER][{strategy_label}] Evaluating {len(files)} frames from {in_dir}")
    metrics = {
        "strategy": strategy_label,
        "n_frames": len(files),
    }

    if len(all_pm_mv) >= 2:
        jitter = compute_static_jitter(
            all_pm_mv,
            all_masks_mv,
            validity_masks_per_frame=all_validity_masks_mv if all_validity_masks_mv else None,
            confidences_per_frame=all_confs_mv if all_confs_mv else None,
            conf_percentile=CONF_PERCENTILE,
            n_anchors=5000,
        )
        if jitter:
            metrics.update(jitter)
            metrics.pop("per_frame_jitter", None)
            _stringify_array_metric(metrics, "per_view_anchor_counts")
    else:
        metrics.update({
            "jitter_mean": np.nan, "jitter_std": np.nan, "jitter_p95": np.nan,
            "jitter_max": np.nan, "drift_mean": np.nan, "hf_jitter": np.nan,
            "n_anchors": 0, "n_potential_anchors": 0,
        })

    timing_path = os.path.join(in_dir, "timing.json")
    if os.path.exists(timing_path):
        try:
            with open(timing_path) as f:
                metrics["align_frames"] = int(json.load(f).get("n_frames", len(files)))
        except Exception:
            pass

    return metrics


def evaluate_strategy_dir(in_dir, strategy_label, dataset_root, dataset_type="dex-ycb"):
    """Evaluate a directory of aligned .npz frames. Returns metrics dict."""
    files = _sorted_frame_paths(in_dir)
    if not files:
        return None

    print(f"    [DEBUG][{strategy_label}] Evaluating {len(files)} frames from {in_dir}")
    t_start_eval = time.perf_counter()

    cham, comp, acc, s_acc, d_acc, s_comp, d_comp = [], [], [], [], [], [], []
    ate_l, rpe_l, rot_l, focal_l, pp_l = [], [], [], [], []
    all_pm_mv, all_masks_mv, all_validity_masks_mv, all_confs_mv = [], [], [], []

    for idx, f in enumerate(files):
        t0_frame = time.perf_counter()
        data = np.load(f)
        gt_pts, est_pts = data["gt_pts"], data["aligned_pts"]

        valid_est = ~np.any(np.isnan(est_pts), axis=-1)
        est_pts = est_pts[valid_est]

        ks, rts, m_2d_raw = data["Ks"], data["R_ts"], data["masks_2d"]

        # masks_2d follows the evaluator contract: True=static, False=dynamic.
        # Hi4D stores person/validity masks instead, so invert only there.
        m_2d_static = ~m_2d_raw if dataset_type == "hi4d" else m_2d_raw

        # Speed optimization: Subsample dense point clouds for metric computation
        MAX_EVAL_PTS = 50000
        if len(est_pts) > MAX_EVAL_PTS:
            est_select = np.random.choice(len(est_pts), MAX_EVAL_PTS, replace=False)
            est_pts = est_pts[est_select]
        if len(gt_pts) > MAX_EVAL_PTS:
            gt_select = np.random.choice(len(gt_pts), MAX_EVAL_PTS, replace=False)
            gt_pts = gt_pts[gt_select]

        if len(est_pts) > 0 and len(gt_pts) > 0:
            # Time specific metrics
            t_metrics_start = time.perf_counter()

            c_dist = compute_chamfer_distance(est_pts, gt_pts)
            cham.append(c_dist)

            c_comp = compute_completeness(est_pts, gt_pts, tau=0.01)
            comp.append(c_comp)

            c_acc = compute_accuracy(est_pts, gt_pts, tau=0.01)
            acc.append(c_acc)

            # Monofusion inputs are generated on DexYCB, so keep static/dynamic split.
            if dataset_type in ("dex-ycb", "monofusion"):
                s_p, d_p = split_points_by_mask(est_pts, m_2d_static, ks, rts)
                g_s, g_d = split_points_by_mask(gt_pts, m_2d_static, ks, rts)
                s_acc.append(compute_accuracy(s_p, g_s, tau=0.01) if len(s_p) > 0 else np.nan)
                d_acc.append(compute_accuracy(d_p, g_d, tau=0.01) if len(d_p) > 0 else np.nan)
                s_comp.append(compute_completeness(s_p, g_s, tau=0.01) if len(g_s) > 0 else np.nan)
                d_comp.append(compute_completeness(d_p, g_d, tau=0.01) if len(g_d) > 0 else np.nan)
            else:
                # For Hi4D, report overall accuracy as "dynamic" for compact summaries.
                d_acc.append(compute_accuracy(est_pts, gt_pts, tau=0.01))
                d_comp.append(c_comp)
                s_acc.append(np.nan)
                s_comp.append(np.nan)

            t_metrics_end = time.perf_counter()
        else:
            for lst in (cham, comp, s_acc, d_acc, s_comp, d_comp):
                lst.append(np.nan)

        s_val, R_val, tr_val = data["scale"], data["R"], data["tr"]

        if "est_poses" in data and data["est_poses"] is not None and data["est_poses"].ndim >= 3:
            e_p, e_i = data["est_poses"], data["est_intrinsics"]
            g_p = np.array(
                [np.linalg.inv(np.vstack([rt, [0, 0, 0, 1]])) if rt.shape == (3, 4) else np.linalg.inv(rt) for rt in
                 rts])
            cam_m = compute_camera_metrics(e_p, g_p, e_i, ks, s_val, R_val, tr_val)
            if not np.isnan(cam_m["ate"]):
                ate_l.append(cam_m["ate"]);
                rpe_l.append(cam_m["rpe"])
                rot_l.append(cam_m["rot_error"]);
                focal_l.append(cam_m["focal_error"])
                pp_l.append(cam_m["pp_error"])

        pm_key = "pointmaps" if "pointmaps" in data else None
        if pm_key:
            V, H, W = normalize_spatial_dims(data)
            pm = normalize_array(data[pm_key], V, H, W)
            conf = normalize_array(data["pointmaps_confs"], V, H, W) if "pointmaps_confs" in data else None
            m_norm = normalize_array(m_2d_static, V, H, W, is_mask=True)
            aligned_pm = np.empty_like(pm)
            for vi in range(V):
                aligned_pm[vi] = apply_similarity_transform(
                    pm[vi].reshape(-1, 3), s_val, R_val, tr_val
                ).reshape(H, W, 3)
            all_pm_mv.append(aligned_pm)
            all_masks_mv.append(m_norm.astype(bool))
            if conf is not None:
                all_confs_mv.append(conf)
            _, vmasks = get_view_names_and_masks(data, dataset_root, dataset_type=dataset_type)
            all_validity_masks_mv.append(np.array([
                vmask if vmask is not None else np.ones((H, W), dtype=bool)
                for vmask in vmasks
            ], dtype=bool))

        t_frame_end = time.perf_counter()
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"      [DEBUG][Frame {idx + 1}/{len(files)}] {t_frame_end - t0_frame:.3f}s (Pts: {len(est_pts)})")

    print(
        f"    [DEBUG][{strategy_label}] Geometric metrics done in {time.perf_counter() - t_start_eval:.2f}s. Computing Jitter...")
    t_jitter_start = time.perf_counter()

    m_s, m_d = np.nanmean(s_acc), np.nanmean(d_acc)
    metrics = {
        "strategy": strategy_label,
        "n_frames": len(files),
        "chamfer": np.nanmean(cham),
        "accuracy": np.nanmean(acc),
        "completeness": np.nanmean(comp),
        "static_comp": np.nanmean(s_comp),
        "dyn_comp": np.nanmean(d_comp),
        "static_acc": m_s,
        "dyn_acc": m_d,
        "motion_gap": m_s - m_d if not np.isnan(m_s) and not np.isnan(m_d) else np.nan,
    }
    if ate_l:
        metrics.update({
            "ate": float(np.nanmean(ate_l)), "rpe": float(np.nanmean(rpe_l)),
            "rot_error": float(np.nanmean(rot_l)), "focal_error": float(np.nanmean(focal_l)),
            "pp_error": float(np.nanmean(pp_l)),
        })
    if len(all_pm_mv) >= 2:
        jitter = compute_static_jitter(
            all_pm_mv,
            all_masks_mv,
            validity_masks_per_frame=all_validity_masks_mv if all_validity_masks_mv else None,
            confidences_per_frame=all_confs_mv if all_confs_mv else None,
            conf_percentile=CONF_PERCENTILE,
            n_anchors=5000,
        )
        if jitter:
            metrics.update(jitter)
            metrics.pop("per_frame_jitter", None)
            _stringify_array_metric(metrics, "per_view_anchor_counts")
            print(f"    [DEBUG][{strategy_label}] Jitter done in {time.perf_counter() - t_jitter_start:.2f}s")

    timing_path = os.path.join(in_dir, "timing.json")
    if os.path.exists(timing_path):
        try:
            with open(timing_path) as f:
                metrics["align_frames"] = int(json.load(f).get("n_frames", len(files)))
        except Exception:
            pass

    return metrics


def add_delta_consistency(df):
    """Δconsistency = Chamfer4D - Chamfer3D (baseline), per view-count."""
    import re
    if df.empty or "strategy" not in df.columns or "chamfer" not in df.columns:
        return df
    df = df.copy()
    df["delta_consistency"] = np.nan

    baseline_by_view = {}
    for _, row in df.iterrows():
        lbl = str(row["strategy"])
        if lbl.startswith("baseline_"):
            m = re.search(r"(\d+views)$", lbl)
            if m:
                baseline_by_view[m.group(1)] = row["chamfer"]

    for idx, row in df.iterrows():
        lbl = str(row["strategy"])
        if lbl.startswith("baseline_"):
            continue
        m = re.search(r"(\d+views)$", lbl)
        if m:
            bl = baseline_by_view.get(m.group(1))
            if bl is not None and not np.isnan(bl):
                df.at[idx, "delta_consistency"] = row["chamfer"] - bl
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GGPT Refinement + Alignment + Evaluation Pipeline")
    parser.add_argument("--subject", type=str, default="all", help="Subject code (e.g. 01) or 'all'")
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--model", type=str, default="vggt", help="Base model being refined")
    parser.add_argument("--dataset", type=str, choices=["dex-ycb", "hi4d", "monofusion"], default="dex-ycb",
                        help="Dataset to use")
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--skip-refinement", action="store_true", help="Skip GGPT refinement, use existing .npz")
    parser.add_argument("--pgo-only", action="store_true", help="Run only Strategy 3 (PGO)")
    parser.add_argument("--base_input_dir", type=str, default=None, help="Base directory for refined inputs")
    parser.add_argument("--jitter", action="store_true", help="Compute and save only jitter/drift metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    refined_model = f"{args.model}-refined"
    dataset_type = args.dataset
    is_4d_model_eval = args.model in ("vggt4d", "monofusion") or dataset_type == "monofusion"
    ds_config = get_dataset_config(dataset_type)
    subject_map = get_subject_by_code(dataset_type)

    # vggt4d uses pair-based directory structure for hi4d:
    # subject-pair00/dance00/Nviews/ instead of subject-dance00/Nviews/
    if args.model == "vggt4d" and dataset_type == "hi4d":
        subject_map = {}
        for name in ds_config["subject_names"]:
            pair, action = name.split("/")
            subject_map[action] = f"subject-{pair}/{action}"

    # Resolve base_input_dir if not provided
    if args.base_input_dir is None:
        if args.model in ["pi3", "pi3x"]:
            if dataset_type == "hi4d":
                args.base_input_dir = os.path.expanduser(f"~/Pi3/ggpt_inputs/hi4d/{args.model}")
            else:
                args.base_input_dir = os.path.expanduser(f"~/Pi3/ggpt_inputs/{dataset_type}/{args.model}")
        elif args.model == "vggt" and dataset_type == "hi4d":
            args.base_input_dir = os.path.expanduser("~/vggt/ggpt_inputs/hi4d")
        elif args.model == "vggt4d":
            if os.path.exists("/local/home/frrajic/xode/fabio/vggt4d_repo/ggpt_inputs"):
                args.base_input_dir = "/local/home/frrajic/xode/fabio/vggt4d_repo/ggpt_inputs"
            else:
                args.base_input_dir = os.path.expanduser(f"~/vggt4d/ggpt_inputs")
        elif dataset_type == "monofusion":
            if os.path.exists("/local/home/frrajic/xode/fabio/monofusion/ggpt_inputs"):
                args.base_input_dir = "/local/home/frrajic/xode/fabio/monofusion/ggpt_inputs"
            else:
                args.base_input_dir = os.path.expanduser("~/monofusion/ggpt_inputs")
        elif dataset_type == "hi4d":
            args.base_input_dir = os.path.expanduser(f"~/{args.model}/ggpt_inputs/hi4d")
        else:
            args.base_input_dir = os.path.expanduser(f"~/{args.model}/ggpt_inputs/{dataset_type}")
    else:
        args.base_input_dir = os.path.abspath(os.path.expanduser(args.base_input_dir))

    print(f"[INFO] Using base_input_dir: {args.base_input_dir}")
    print(f"[INFO] Dataset: {dataset_type}")

    # Resolve subjects
    if args.subject == "all":
        if dataset_type == "monofusion":
            subj_folders = sorted(glob.glob(os.path.join(args.base_input_dir, "subject-*")))
            subjects = [os.path.basename(f).replace("subject-", "") for f in subj_folders]
            subject_map.update({subj: f"subject-{subj}" for subj in subjects})
        else:
            subjects = sorted(subject_map.keys())
    else:
        # If the user passed e.g. pair00/dance00, extract dance00
        subj_code = args.subject.split("/")[-1]
        if subj_code.startswith("subject-"):
            subj_code = subj_code.replace("subject-", "", 1)
        subjects = [subj_code]

    # Resolve Hi4D alternative pair folder naming (e.g. hug09 -> pair09_hug09) if needed
    if dataset_type == "hi4d":
        from models.scripts.eval_config import get_pair_name_for_subject
        resolved_subjects = []
        for subj in subjects:
            subject_full = subject_map.get(subj, f"subject-{subj}")
            scene_dir_default = os.path.join(args.base_input_dir, subject_full)
            if not os.path.isdir(scene_dir_default):
                pair_name = get_pair_name_for_subject(dataset_type, subject_full)
                if pair_name:
                    alt_subject_full = f"subject-{pair_name}_{subj}"
                    if os.path.isdir(os.path.join(args.base_input_dir, alt_subject_full)):
                        resolved_subjects.append(f"{pair_name}_{subj}")
                        subject_map[f"{pair_name}_{subj}"] = alt_subject_full
                        continue
            resolved_subjects.append(subj)
        subjects = resolved_subjects
    # Step 0: Run GGPT refinement if needed
    if not args.skip_refinement:
        for subj in subjects:
            for nv in args.views:
                print(f"\n[STAGE] Running GGPT refinement: subject={subj} views={nv} model={args.model}")
                subject_full = subject_map.get(subj, f"subject-{subj}")
                scene_dir = os.path.join(args.base_input_dir, subject_full, f"{nv}views")

                if not os.path.isdir(scene_dir):
                    # Check if clean name style exists (e.g. subject-10 instead of full DexYCB name)
                    alt_subject_full = f"subject-{subj}"
                    alt_scene_dir = os.path.join(args.base_input_dir, alt_subject_full, f"{nv}views")
                    if os.path.isdir(alt_scene_dir):
                        scene_dir = alt_scene_dir
                    else:
                        print(f"  [WARN] Input directory not found: {scene_dir}. Skipping.")
                        continue

                baseline_dir = os.path.join("aligned_outputs", refined_model, "baseline", subject_full, f"{nv}views")
                existing = _sorted_frame_paths(baseline_dir)
                if existing:
                    print(f"  [SKIP] Refined frames already exist for {subj}/{nv}views ({len(existing)} frames)")
                    continue

                refinement_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_ggpt_refinement.py")
                cmd = [
                    sys.executable, refinement_script,
                    "--subject", subj,
                    "--views", str(nv),
                    "--model", args.model,
                    "--dataset", dataset_type,
                    "--base_input_dir", args.base_input_dir,
                ]
                if args.no_rerun:
                    cmd.append("--no_rerun")
                print(f"  CMD: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

    # Step 1-3: Alignment + Evaluation per subject
    all_subject_dfs = []

    for subj in subjects:
        subject_full = subject_map.get(subj)
        if not subject_full:
            print(f"[WARN] Unknown subject code: {subj}")
            continue

        csv_suffix = "_jitter" if args.jitter else ""
        csv_path = f"eval_summary_{refined_model}_{subj}{csv_suffix}.csv"
        if os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} already exists, skipping evaluation for subject {subj}.")
            existing_df = pd.read_csv(csv_path)
            all_subject_dfs.append(existing_df)
            continue

        dataset_root = get_dataset_root_for_subject(dataset_type, subject_full)
        if dataset_type == "monofusion":
            dataset_root = os.path.join(args.base_input_dir, subject_full)
        if dataset_type != "monofusion" and not os.path.isdir(dataset_root):
            print(f"[WARN] Dataset not found: {dataset_root}")
            continue

        subject_results = []

        for nv in args.views:
            raw_dir = os.path.join("aligned_outputs", refined_model, "baseline", subject_full, f"{nv}views")
            raw_paths = _sorted_frame_paths(raw_dir)
            if len(raw_paths) < 2:
                print(f"[WARN] Not enough frames for {subj}/{nv}views (found {len(raw_paths)})")
                continue

            # Rerun
            if not args.no_rerun and rr is not None:
                rerun_eye = ds_config.get("eye_up", RERUN_EYE_UP)
                init_recording(subj, nv, model_name=refined_model)
                configure_rerun_view_defaults("world", rerun_eye)

            # ── Baseline (per-frame Umeyama) ──────────────────────────────
            baseline_out = os.path.join("aligned_outputs", refined_model, "baseline_aligned", subject_full,
                                        f"{nv}views")
            if not args.pgo_only:
                print(f"\n[STAGE] Baseline alignment: {subj}/{nv}views")
                baseline_paths = run_baseline_alignment(raw_paths, dataset_root, baseline_out,
                                                        dataset_type=dataset_type)
            else:
                baseline_paths = _sorted_frame_paths(baseline_out)

            if not args.no_rerun and rr is not None:
                print(f"  [RERUN] Logging GT and Baseline...")
                try:
                    # Filter paths to ensure we don't pass None to join inside these helpers
                    valid_baseline_paths = []
                    for p in baseline_paths:
                        d = np.load(p)
                        v_names, _ = get_view_names_and_masks(d, dataset_root, dataset_type=dataset_type)
                        if any(v is not None for v in v_names):
                            valid_baseline_paths.append(p)

                    if valid_baseline_paths:
                        log_gt_sequence(valid_baseline_paths, dataset_root, dataset_type=dataset_type, log_root="world")

                        baseline_tfs = []
                        for p in valid_baseline_paths:
                            d = np.load(p)
                            baseline_tfs.append((d["scale"], d["R"], d["tr"]))

                        log_aligned_sequence(valid_baseline_paths, baseline_tfs, 1.0, np.eye(3), np.zeros(3),
                                             "Baseline", [0, 0, 250], dataset_root, dataset_type=dataset_type,
                                             log_root="world")
                except Exception as e:
                    print(f"  [RERUN][WARN] GT/Baseline logging failed: {e}")

            if len(baseline_paths) < 2:
                print(f"[WARN] Baseline produced too few frames for {subj}/{nv}views")
                continue

            # Evaluate baseline
            print(f"  [EVAL] Starting baseline evaluation for {subj}/{nv}views...")
            if args.jitter:
                bl_metrics = evaluate_jitter_strategy_dir(
                    baseline_out, f"baseline_{nv}views", dataset_root, dataset_type=dataset_type,
                )
            else:
                bl_metrics = evaluate_strategy_dir(
                    baseline_out, f"baseline_{nv}views", dataset_root, dataset_type=dataset_type,
                )
            if bl_metrics:
                print(f"  [EVAL] Baseline evaluation complete.")
                bl_metrics["subject"] = subject_full
                subject_results.append(bl_metrics)

            if is_4d_model_eval:
                # ── Global Alignment (One global Umeyama for the full sequence) ──
                global_out = os.path.join("aligned_outputs", refined_model, "global", subject_full, f"{nv}views")
                print(f"\n[STAGE] Global alignment: {subj}/{nv}views")

                # For global alignment, frame transforms are identity, and we solve one global registration transform
                def identity_strategy_func(paths, root, dataset_type):
                    return [(1.0, np.eye(3), np.zeros(3)) for _ in paths]

                run_strategy_alignment(baseline_paths, dataset_root, identity_strategy_func, global_out,
                                       f"global_{nv}views", dataset_type=dataset_type)

                print(f"  [EVAL] Evaluating Global...")
                if args.jitter:
                    glob_metrics = evaluate_jitter_strategy_dir(
                        global_out, f"global_{nv}views", dataset_root, dataset_type=dataset_type,
                    )
                else:
                    glob_metrics = evaluate_strategy_dir(
                        global_out, f"global_{nv}views", dataset_root, dataset_type=dataset_type,
                    )
                if glob_metrics:
                    glob_metrics["subject"] = subject_full
                    subject_results.append(glob_metrics)

                if not args.no_rerun and rr is not None:
                    try:
                        tf_glob = identity_strategy_func(baseline_paths, dataset_root, dataset_type=dataset_type)
                        s_glob, R_glob, tr_glob = solve_final_gt_registration(baseline_paths, tf_glob, dataset_root,
                                                                              use_static_mask=False,
                                                                              dataset_type=dataset_type)
                        log_aligned_sequence(baseline_paths, tf_glob, s_glob, R_glob, tr_glob,
                                             "Global", [255, 0, 255], dataset_root, dataset_type=dataset_type,
                                             log_root="world")
                    except Exception as e:
                        print(f"  [RERUN][WARN] Global logging failed: {e}")
            else:
                # ── Strategy 1 (Reference) ────────────────────────────────────
                if not args.pgo_only:
                    s1_dir = os.path.join("aligned_outputs", refined_model, "strategy1", subject_full, f"{nv}views")
                    print(f"\n[STAGE] Strategy 1: {subj}/{nv}views")
                    run_strategy_alignment(baseline_paths, dataset_root, strategy1_reference, s1_dir,
                                           f"strategy1_{nv}views", dataset_type=dataset_type)
                    print(f"  [EVAL] Evaluating Strategy 1...")
                    if args.jitter:
                        s1_metrics = evaluate_jitter_strategy_dir(
                            s1_dir, f"strategy1_{nv}views", dataset_root, dataset_type=dataset_type,
                        )
                    else:
                        s1_metrics = evaluate_strategy_dir(
                            s1_dir, f"strategy1_{nv}views", dataset_root, dataset_type=dataset_type,
                        )
                    if s1_metrics:
                        s1_metrics["subject"] = subject_full
                        subject_results.append(s1_metrics)

                    if not args.no_rerun and rr is not None:
                        try:
                            tf_s1 = strategy1_reference(baseline_paths, dataset_root, dataset_type=dataset_type)
                            s_g1, R_g1, tr_g1 = solve_final_gt_registration(baseline_paths, tf_s1, dataset_root,
                                                                            use_static_mask=False,
                                                                            dataset_type=dataset_type)
                            log_aligned_sequence(baseline_paths, tf_s1, s_g1, R_g1, tr_g1,
                                                 "Strategy_1", [255, 0, 0], dataset_root, dataset_type=dataset_type,
                                                 log_root="world")
                        except Exception as e:
                            print(f"  [RERUN][WARN] Strategy 1 logging failed: {e}")

                # ── Strategy 2 (Hierarchical) ─────────────────────────────────
                if not args.pgo_only:
                    s2_dir = os.path.join("aligned_outputs", refined_model, "strategy2", subject_full, f"{nv}views")
                    print(f"\n[STAGE] Strategy 2: {subj}/{nv}views")
                    run_strategy_alignment(baseline_paths, dataset_root, strategy2_hierarchical, s2_dir,
                                           f"strategy2_{nv}views", dataset_type=dataset_type)
                    if args.jitter:
                        s2_metrics = evaluate_jitter_strategy_dir(
                            s2_dir, f"strategy2_{nv}views", dataset_root, dataset_type=dataset_type,
                        )
                    else:
                        s2_metrics = evaluate_strategy_dir(
                            s2_dir, f"strategy2_{nv}views", dataset_root, dataset_type=dataset_type,
                        )
                    if s2_metrics:
                        s2_metrics["subject"] = subject_full
                        subject_results.append(s2_metrics)

                    if not args.no_rerun and rr is not None:
                        try:
                            tf_s2 = strategy2_hierarchical(baseline_paths, dataset_root, dataset_type=dataset_type)
                            s_g2, R_g2, tr_g2 = solve_final_gt_registration(baseline_paths, tf_s2, dataset_root,
                                                                            use_static_mask=False,
                                                                            dataset_type=dataset_type)
                            log_aligned_sequence(baseline_paths, tf_s2, s_g2, R_g2, tr_g2,
                                                 "Strategy_2", [255, 0, 255], dataset_root, dataset_type=dataset_type,
                                                 log_root="world")
                        except Exception as e:
                            print(f"  [RERUN][WARN] Strategy 2 logging failed: {e}")

                # ── Strategy 3 (PGO) ──────────────────────────────────────────
                s3_dir = os.path.join("aligned_outputs", refined_model, "strategy3", subject_full, f"{nv}views")
                print(f"\n[STAGE] Strategy 3 (PGO): {subj}/{nv}views")
                run_strategy_alignment(baseline_paths, dataset_root, strategy3_pgo, s3_dir,
                                       f"strategy3_{nv}views", dataset_type=dataset_type, num_iters=50)
                if args.jitter:
                    s3_metrics = evaluate_jitter_strategy_dir(
                        s3_dir, f"strategy3_{nv}views", dataset_root, dataset_type=dataset_type,
                    )
                else:
                    s3_metrics = evaluate_strategy_dir(
                        s3_dir, f"strategy3_{nv}views", dataset_root, dataset_type=dataset_type,
                    )
                if s3_metrics:
                    s3_metrics["subject"] = subject_full
                    subject_results.append(s3_metrics)

                if not args.no_rerun and rr is not None:
                    try:
                        tf_s3 = strategy3_pgo(baseline_paths, dataset_root, num_iters=50, dataset_type=dataset_type)
                        s_g3, R_g3, tr_g3 = solve_final_gt_registration(baseline_paths, tf_s3, dataset_root,
                                                                        use_static_mask=False,
                                                                        dataset_type=dataset_type)
                        log_aligned_sequence(baseline_paths, tf_s3, s_g3, R_g3, tr_g3,
                                             "Strategy_3_PGO", [0, 150, 150], dataset_root, dataset_type=dataset_type,
                                             log_root="world")
                    except Exception as e:
                        print(f"  [RERUN][WARN] Strategy 3 logging failed: {e}")

        # Save per-subject CSV
        if subject_results:
            df = add_delta_consistency(pd.DataFrame(subject_results))
            print(f"\n=== Performance Summary: {subject_full} ({refined_model}) ===")
            pd.set_option("display.precision", 5)
            pd.set_option("display.width", 2000)
            pd.set_option("display.max_columns", None)
            if dataset_type == "hi4d":
                cols = [
                    "strategy", "n_frames", "chamfer", "delta_consistency", "accuracy", "completeness",
                    "ate", "rpe", "rot_error", "jitter_mean", "drift_mean", "hf_jitter"
                ]
            else:
                cols = [c for c in [
                    "strategy", "n_frames", "chamfer", "delta_consistency", "accuracy", "completeness",
                    "static_comp", "dyn_comp", "static_acc", "dyn_acc", "motion_gap",
                    "ate", "rpe", "rot_error", "focal_error", "pp_error",
                    "jitter_mean", "jitter_std", "jitter_p95", "jitter_max",
                    "drift_mean", "hf_jitter",
                ] if c in df.columns]
            print(df[cols].to_string(index=False))
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Saved per-subject report: {csv_path}")
            all_subject_dfs.append(df)

    # ── Aggregate all subjects ────────────────────────────────────────────
    if all_subject_dfs:
        combined = pd.concat(all_subject_dfs, ignore_index=True)
        numeric = combined.select_dtypes(include="number").copy()
        numeric["strategy"] = combined["strategy"]
        agg = numeric.groupby("strategy").mean().reset_index().sort_values("strategy")

        print("\n" + "=" * 80)
        print(f"CROSS-SUBJECT AGGREGATED RESULTS ({refined_model})")
        print("=" * 80)
        if dataset_type == "hi4d":
            cols = [
                "strategy", "n_frames", "chamfer", "delta_consistency", "accuracy", "completeness",
                "ate", "rpe", "rot_error", "jitter_mean", "jitter_std", "jitter_p95", "jitter_max", "drift_mean",
                "hf_jitter"
            ]
        else:
            cols = [c for c in [
                "strategy", "n_frames", "chamfer", "delta_consistency", "accuracy", "completeness",
                "static_comp", "dyn_comp", "static_acc", "dyn_acc", "motion_gap",
                "ate", "rpe", "rot_error", "jitter_mean", "jitter_std", "jitter_p95", "jitter_max", "drift_mean",
                "hf_jitter",
            ] if c in agg.columns]
        print(agg[cols].to_string(index=False))

        csv_suffix = "_jitter" if args.jitter else ""
        agg_path = f"eval_summary_ALL_{refined_model}{csv_suffix}.csv"
        agg.to_csv(agg_path, index=False)
        print(f"\n[INFO] Aggregated results saved to {agg_path}")
    else:
        print("[WARN] No results to aggregate.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
