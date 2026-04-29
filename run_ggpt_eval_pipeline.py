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
)
from utils.umeyama_alignment import estimate_similarity_transform, apply_similarity_transform
from utils.alignment_4D import (
    strategy1_reference, strategy2_hierarchical, strategy3_pgo,
    solve_final_gt_registration, normalize_spatial_dims, normalize_array,
    extract_clean_gt_correspondences, get_view_names_and_masks,
)
from utils.camera_utils import discover_view_name
from utils.gt import build_gt_validity_masks
from utils.temporal_metrics import (
    compute_chamfer_distance, compute_accuracy, compute_completeness,
    split_points_by_mask, compute_static_jitter, compute_camera_metrics,
)

try:
    import rerun as rr
    from utils.rerun_logging import (
        init_recording, configure_rerun_view_defaults,
        log_pointcloud, log_gt_sequence, log_aligned_sequence,
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

def run_baseline_alignment(frame_paths, dataset_root, out_dir, skip_existing=True):
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

        view_names, vmasks = get_view_names_and_masks(data, dataset_root)
        if all(m is None for m in vmasks):
            print(f"    [WARN] Frame {t}: Could not resolve any view names, skipping.")
            continue

        # Build correspondences for Umeyama (using already-filtered points)
        res = extract_clean_gt_correspondences(data, dataset_root, use_static_mask=False)
        if res is None:
            print(f"    [WARN] Frame {t}: No correspondences found, skipping.")
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
        gt_pts = data["gt_pts"]
        if gt_pts.ndim == 4:
            # (V, H, W, 3) -> flatten valid
            gt_mask_all = np.ones(gt_pts.shape[:3], dtype=bool)
            for v in range(V):
                if vmasks[v] is not None:
                    gt_mask_all[v] &= vmasks[v]
            gt_pts = gt_pts[gt_mask_all]
        if np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
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
                           strategy_label, skip_existing=True, **kwargs):
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
    frame_transforms = strategy_func(baseline_paths, dataset_root, **kwargs)
    s_glob, R_glob, tr_glob = solve_final_gt_registration(
        baseline_paths, frame_transforms, dataset_root, use_static_mask=False
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
        view_names = [discover_view_name(dataset_root, k) for k in ks]
        vmasks = build_gt_validity_masks(t, view_names, dataset_root, target_hw=(H, W))

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
        if np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
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

def evaluate_strategy_dir(in_dir, strategy_label):
    """Evaluate a directory of aligned .npz frames. Returns metrics dict."""
    files = _sorted_frame_paths(in_dir)
    if not files:
        return None

    cham, comp, s_acc, d_acc, s_comp, d_comp = [], [], [], [], [], []
    ate_l, rpe_l, rot_l, focal_l, pp_l = [], [], [], [], []
    all_pm_mv, all_masks_mv = [], []

    for f in files:
        data = np.load(f)
        gt_pts, est_pts = data["gt_pts"], data["aligned_pts"]

        valid_est = ~np.any(np.isnan(est_pts), axis=-1)
        est_pts = est_pts[valid_est]

        ks, rts, m_2d = data["Ks"], data["R_ts"], data["masks_2d"]

        if len(est_pts) > 0 and len(gt_pts) > 0:
            cham.append(compute_chamfer_distance(est_pts, gt_pts))
            comp.append(compute_completeness(est_pts, gt_pts, tau=0.01))
            s_p, d_p = split_points_by_mask(est_pts, m_2d, ks, rts)
            g_s, g_d = split_points_by_mask(gt_pts, m_2d, ks, rts)
            s_acc.append(compute_accuracy(s_p, g_s, tau=0.01) if len(s_p) > 0 else np.nan)
            d_acc.append(compute_accuracy(d_p, g_d, tau=0.01) if len(d_p) > 0 else np.nan)
            s_comp.append(compute_completeness(s_p, g_s, tau=0.01) if len(g_s) > 0 else np.nan)
            d_comp.append(compute_completeness(d_p, g_d, tau=0.01) if len(g_d) > 0 else np.nan)
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
            m_norm = normalize_array(m_2d, V, H, W, is_mask=True)
            aligned_pm = np.empty_like(pm)
            for vi in range(V):
                aligned_pm[vi] = apply_similarity_transform(
                    pm[vi].reshape(-1, 3), s_val, R_val, tr_val
                ).reshape(H, W, 3)
            all_pm_mv.append(aligned_pm)
            all_masks_mv.append(m_norm.astype(bool))

    m_s, m_d = np.nanmean(s_acc), np.nanmean(d_acc)
    metrics = {
        "strategy": strategy_label,
        "n_frames": len(files),
        "chamfer": np.nanmean(cham),
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
        jitter = compute_static_jitter(all_pm_mv, all_masks_mv, n_anchors=5000)
        if jitter:
            metrics.update(jitter)

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
    if df.empty or "strategy" not in df.columns:
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
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--skip-refinement", action="store_true", help="Skip GGPT refinement, use existing .npz")
    parser.add_argument("--pgo-only", action="store_true", help="Run only Strategy 3 (PGO)")
    parser.add_argument("--base_input_dir", type=str, default=None, help="Base directory for refined inputs")
    return parser.parse_args()


def main():
    args = parse_args()
    refined_model = f"{args.model}-refined"

    # Resolve base_input_dir if not provided
    if args.base_input_dir is None:
        # vggt -> ~/vggt/ggpt_inputs
        # mast3r -> ~/mast3r/ggpt_inputs
        # pi3 -> ~/Pi3/ggpt_inputs/pi3
        # pi3x -> ~/Pi3/ggpt_inputs/pi3x
        if args.model in ["pi3", "pi3x"]:
            args.base_input_dir = os.path.expanduser(f"~/Pi3/ggpt_inputs/{args.model}")
        else:
            args.base_input_dir = os.path.expanduser(f"~/{args.model}/ggpt_inputs")
        print(f"[INFO] Inferred base_input_dir: {args.base_input_dir}")

    # Resolve subjects
    if args.subject == "all":
        subjects = sorted(SUBJECT_BY_CODE.keys())
    else:
        subjects = [args.subject]

    # Step 0: Run GGPT refinement if needed
    if not args.skip_refinement:
        for subj in subjects:
            for nv in args.views:
                baseline_dir = os.path.join("aligned_outputs", refined_model, "baseline",
                                            SUBJECT_BY_CODE.get(subj, f"subject-{subj}"), f"{nv}views")
                existing = _sorted_frame_paths(baseline_dir)
                if existing:
                    print(f"[SKIP] Refined frames already exist for {subj}/{nv}views ({len(existing)} frames)")
                    continue

                print(f"\n[STAGE] Running GGPT refinement: subject={subj} views={nv} model={args.model}")
                refinement_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_ggpt_refinement.py")
                cmd = [
                    sys.executable, refinement_script,
                    "--subject", subj,
                    "--views", str(nv),
                    "--model", args.model,
                    "--base_input_dir", args.base_input_dir,
                ]
                if args.no_rerun:
                    cmd.append("--no_rerun")
                print(f"  CMD: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

    # Step 1-3: Alignment + Evaluation per subject
    all_subject_dfs = []

    for subj in subjects:
        subject_full = SUBJECT_BY_CODE.get(subj)
        if not subject_full:
            print(f"[WARN] Unknown subject code: {subj}")
            continue

        csv_path = f"eval_summary_{refined_model}_{subj}.csv"
        if os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} already exists, skipping evaluation for subject {subj}.")
            existing_df = pd.read_csv(csv_path)
            all_subject_dfs.append(existing_df)
            continue

        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_full)
        if not os.path.isdir(dataset_root):
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
                init_recording(subj, nv, model_name=refined_model)
                configure_rerun_view_defaults("world", RERUN_EYE_UP)

            # ── Baseline (per-frame Umeyama) ──────────────────────────────
            baseline_out = os.path.join("aligned_outputs", refined_model, "baseline_aligned", subject_full,
                                        f"{nv}views")
            if not args.pgo_only:
                print(f"\n[STAGE] Baseline alignment: {subj}/{nv}views")
                baseline_paths = run_baseline_alignment(raw_paths, dataset_root, baseline_out)
            else:
                baseline_paths = _sorted_frame_paths(baseline_out)

            if not args.no_rerun and rr is not None:
                print(f"  [RERUN] Logging GT and Baseline...")
                try:
                    log_gt_sequence(baseline_paths, dataset_root, log_root="world")
                    # For baseline, we use the per-frame Umeyama transforms saved in each .npz
                    baseline_tfs = []
                    for p in baseline_paths:
                        d = np.load(p)
                        # We reconstruct the (s, R, tr) from the saved metadata
                        baseline_tfs.append((d["scale"], d["R"], d["tr"]))

                    log_aligned_sequence(baseline_paths, baseline_tfs, 1.0, np.eye(3), np.zeros(3),
                                         "Baseline", [0, 0, 250], dataset_root, log_root="world")
                except Exception as e:
                    print(f"  [RERUN][WARN] GT/Baseline logging failed: {e}")

            if len(baseline_paths) < 2:
                print(f"[WARN] Baseline produced too few frames for {subj}/{nv}views")
                continue

            # Evaluate baseline
            bl_metrics = evaluate_strategy_dir(baseline_out, f"baseline_{nv}views")
            if bl_metrics:
                bl_metrics["subject"] = subject_full
                subject_results.append(bl_metrics)

            # ── Strategy 1 (Reference) ────────────────────────────────────
            if not args.pgo_only:
                s1_dir = os.path.join("aligned_outputs", refined_model, "strategy1", subject_full, f"{nv}views")
                print(f"\n[STAGE] Strategy 1: {subj}/{nv}views")
                run_strategy_alignment(baseline_paths, dataset_root, strategy1_reference, s1_dir,
                                       f"strategy1_{nv}views")
                s1_metrics = evaluate_strategy_dir(s1_dir, f"strategy1_{nv}views")
                if s1_metrics:
                    s1_metrics["subject"] = subject_full
                    subject_results.append(s1_metrics)

                if not args.no_rerun and rr is not None:
                    try:
                        tf_s1 = strategy1_reference(baseline_paths, dataset_root)
                        s_g1, R_g1, tr_g1 = solve_final_gt_registration(baseline_paths, tf_s1, dataset_root,
                                                                        use_static_mask=False)
                        log_aligned_sequence(baseline_paths, tf_s1, s_g1, R_g1, tr_g1,
                                             "Strategy_1", [255, 0, 0], dataset_root, log_root="world")
                    except Exception as e:
                        print(f"  [RERUN][WARN] Strategy 1 logging failed: {e}")

            # ── Strategy 2 (Hierarchical) ─────────────────────────────────
            if not args.pgo_only:
                s2_dir = os.path.join("aligned_outputs", refined_model, "strategy2", subject_full, f"{nv}views")
                print(f"\n[STAGE] Strategy 2: {subj}/{nv}views")
                run_strategy_alignment(baseline_paths, dataset_root, strategy2_hierarchical, s2_dir,
                                       f"strategy2_{nv}views")
                s2_metrics = evaluate_strategy_dir(s2_dir, f"strategy2_{nv}views")
                if s2_metrics:
                    s2_metrics["subject"] = subject_full
                    subject_results.append(s2_metrics)

                if not args.no_rerun and rr is not None:
                    try:
                        tf_s2 = strategy2_hierarchical(baseline_paths, dataset_root)
                        s_g2, R_g2, tr_g2 = solve_final_gt_registration(baseline_paths, tf_s2, dataset_root,
                                                                        use_static_mask=False)
                        log_aligned_sequence(baseline_paths, tf_s2, s_g2, R_g2, tr_g2,
                                             "Strategy_2", [255, 0, 255], dataset_root, log_root="world")
                    except Exception as e:
                        print(f"  [RERUN][WARN] Strategy 2 logging failed: {e}")

            # ── Strategy 3 (PGO) ──────────────────────────────────────────
            s3_dir = os.path.join("aligned_outputs", refined_model, "strategy3", subject_full, f"{nv}views")
            print(f"\n[STAGE] Strategy 3 (PGO): {subj}/{nv}views")
            run_strategy_alignment(baseline_paths, dataset_root, strategy3_pgo, s3_dir,
                                   f"strategy3_{nv}views", num_iters=50)
            s3_metrics = evaluate_strategy_dir(s3_dir, f"strategy3_{nv}views")
            if s3_metrics:
                s3_metrics["subject"] = subject_full
                subject_results.append(s3_metrics)

            if not args.no_rerun and rr is not None:
                try:
                    tf_s3 = strategy3_pgo(baseline_paths, dataset_root, num_iters=50)
                    s_g3, R_g3, tr_g3 = solve_final_gt_registration(baseline_paths, tf_s3, dataset_root,
                                                                    use_static_mask=False)
                    log_aligned_sequence(baseline_paths, tf_s3, s_g3, R_g3, tr_g3,
                                         "Strategy_3_PGO", [0, 150, 150], dataset_root, log_root="world")
                except Exception as e:
                    print(f"  [RERUN][WARN] Strategy 3 logging failed: {e}")

        # Save per-subject CSV
        if subject_results:
            df = add_delta_consistency(pd.DataFrame(subject_results))
            print(f"\n=== Performance Summary: {subject_full} ({refined_model}) ===")
            pd.set_option("display.precision", 5)
            pd.set_option("display.width", 2000)
            pd.set_option("display.max_columns", None)
            cols = [c for c in [
                "strategy", "n_frames", "chamfer", "delta_consistency", "completeness",
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
        cols = [c for c in [
            "strategy", "n_frames", "chamfer", "delta_consistency", "completeness",
            "static_comp", "dyn_comp", "static_acc", "dyn_acc", "motion_gap",
            "ate", "rpe", "rot_error", "jitter_mean", "drift_mean", "hf_jitter",
        ] if c in agg.columns]
        print(agg[cols].to_string(index=False))

        agg_path = f"eval_summary_ALL_{refined_model}.csv"
        agg.to_csv(agg_path, index=False)
        print(f"\n[INFO] Aggregated results saved to {agg_path}")
    else:
        print("[WARN] No results to aggregate.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
