#!/usr/bin/env python3
"""
VGGT4D Native 4D Evaluation — Dual-Tier Alignment.

For each (subject, view_count) generates exactly TWO rows of metrics
from the SAME model output (baseline NPZ files):

┌────────────────────────────────────────────────────────────────────────┐
│  Row 1 — baseline_Nviews   (Per-Frame Umeyama)                       │
│    Uses the pre-aligned `aligned_pts` already stored in each NPZ.    │
│    Best-case spatial accuracy — each frame gets its own (s,R,t).     │
│                                                                      │
│  Row 2 — global_Nviews     (One Global Umeyama for the full seq)     │
│    Computes ONE Umeyama across ALL frames/views, then applies it     │
│    everywhere.  Measures the model's native 4D consistency.          │
│                                                                      │
│  delta_consistency = chamfer_4d(global) − chamfer_3d(baseline)       │
│    Should be near zero for a temporally-stable model.                │
└────────────────────────────────────────────────────────────────────────┘

All other metrics (ATE, RPE, rot_error, focal_error, pp_error,
jitter, completeness, static/dyn accuracy) are computed for BOTH rows.
"""
import os
import csv
import glob
import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from vggt.utils.umeyama_alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
)
from vggt.utils.temporal_metrics import (
    compute_chamfer_distance,
    compute_accuracy,
    compute_completeness,
    split_points_by_mask,
    compute_static_jitter,
    compute_camera_metrics,
)
from vggt.utils.alignment_4d import normalize_spatial_dims, normalize_array
from vggt.utils.gt import (
    get_single_view_correspondences,
    build_gt_validity_masks,
    DEPTH_MAX_M,
)
from vggt.utils.camera_utils import discover_view_name
from vggt.utils.alignment_4d import (
    normalize_spatial_dims, normalize_array,
    strategy1_reference, strategy2_hierarchical, strategy3_pgo,
    solve_final_gt_registration,
)
from eval_config import (
    SUBJECT_NAMES,
    SUBJECT_BY_CODE,
    DATASET_BASE_ROOT,
    CONF_PERCENTILE,
    RERUN_ADDR,
    DATASETS,
    get_dataset_config,
    get_subject_by_code,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: compute all metrics for a set of frame NPZs under a specific alignment
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_metrics_for_alignment(
        files, strategy_label, dataset_root,
        global_transform=None,
        per_frame_transforms=None,
        gt_registration=None,
        out_plot_dir=None,
        dataset_type="dex-ycb",
):
    """
    Compute the full metric suite for one alignment mode.
    """
    if not files:
        return None

    first_data = np.load(files[0], allow_pickle=True)
    V, H, W = normalize_spatial_dims(first_data)
    ks0 = first_data["Ks"]
    view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in ks0]

    cham_dist, comp_score, acc_score = [], [], []
    s_acc_list, d_acc_list = [], []
    s_comp_list, d_comp_list = [], []
    ate_list, rpe_list, rot_err_list, focal_err_list, pp_err_list = [], [], [], [], []
    all_pointmaps_mv, all_masks_mv = [], []

    for f in files:
        data = np.load(f, allow_pickle=True)
        gt_pts = data["gt_pts"]
        ks, rts = data["Ks"], data["R_ts"]
        m_2d = data.get("masks_2d")

        # ── Obtain aligned point cloud ───────────────────────────────────
        if per_frame_transforms is not None and gt_registration is not None:
            # Strategy mode: apply per-frame transform then global GT registration
            f_idx = files.index(f)
            s_i, R_i, tr_i = per_frame_transforms[f_idx]
            s_g, R_g, tr_g = gt_registration
            # Combined: T_gt ∘ T_i  →  s_tot = s_g*s_i, R_tot = R_g@R_i, tr_tot = s_g*(R_g@tr_i)+tr_g
            s_val = s_g * s_i
            R_val = R_g @ R_i
            tr_val = s_g * (R_g @ tr_i) + tr_g

            pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
            conf = normalize_array(data["pointmaps_confs"], V, H, W)
            t_idx = int(data["frame_idx"])

            # ── Global threshold for the whole frame ──
            frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE) if conf is not None else 0.0

            vmasks = build_gt_validity_masks(
                t_idx, view_names, dataset_root,
                depth_max_m=DEPTH_MAX_M, target_hw=(H, W),
                dataset_type=dataset_type,
            )

            parts = []
            for v in range(V):
                pts_flat = pm[v].reshape(-1, 3)
                valid = conf[v].ravel() > frame_thr
                if vmasks[v] is not None:
                    vm = vmasks[v]
                    if vm.shape != (H, W):
                        vm = cv2.resize(
                            vm.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    valid &= vm.ravel()
                if valid.any():
                    parts.append(
                        apply_similarity_transform(pts_flat[valid], s_val, R_val, tr_val)
                    )

            est_pts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))

        elif global_transform is not None:
            # Global mode: re-align raw pointmaps with the single transform
            s_val, R_val, tr_val = global_transform
            pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
            conf = normalize_array(data["pointmaps_confs"], V, H, W)
            t_idx = int(data["frame_idx"])

            # ── Global threshold for the whole frame ──
            frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE) if conf is not None else 0.0

            vmasks = build_gt_validity_masks(
                t_idx, view_names, dataset_root,
                depth_max_m=DEPTH_MAX_M, target_hw=(H, W),
                dataset_type=dataset_type,
            )

            parts = []
            for v in range(V):
                pts_flat = pm[v].reshape(-1, 3)
                valid = conf[v].ravel() > frame_thr
                if vmasks[v] is not None:
                    vm = vmasks[v]
                    if vm.shape != (H, W):
                        vm = cv2.resize(
                            vm.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    valid &= vm.ravel()
                if valid.any():
                    parts.append(
                        apply_similarity_transform(pts_flat[valid], s_val, R_val, tr_val)
                    )

            est_pts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))
        else:
            # Baseline mode: use pre-aligned points
            est_pts = data["aligned_pts"]
            s_val, R_val, tr_val = float(data["scale"]), data["R"], data["tr"]

        # Clean NaNs
        valid_est = ~np.any(np.isnan(est_pts), axis=-1)
        est_pts = est_pts[valid_est]

        # ── Chamfer & accuracy ───────────────────────────────────────────
        if len(est_pts) > 0 and len(gt_pts) > 0:
            cham_dist.append(compute_chamfer_distance(est_pts, gt_pts))
            comp_score.append(compute_completeness(est_pts, gt_pts, tau=0.01))
            acc_score.append(compute_accuracy(est_pts, gt_pts, tau=0.01))

            if dataset_type == "dex-ycb" and m_2d is not None:
                s_p, d_p = split_points_by_mask(est_pts, m_2d, ks, rts)
                g_s, g_d = split_points_by_mask(gt_pts, m_2d, ks, rts)

                s_acc_list.append(compute_accuracy(s_p, g_s, tau=0.01) if len(s_p) > 0 else np.nan)
                d_acc_list.append(compute_accuracy(d_p, g_d, tau=0.01) if len(d_p) > 0 else np.nan)
                s_comp_list.append(compute_completeness(s_p, g_s, tau=0.01) if len(g_s) > 0 else np.nan)
                d_comp_list.append(compute_completeness(d_p, g_d, tau=0.01) if len(g_d) > 0 else np.nan)
        else:
            cham_dist.append(np.nan)
            comp_score.append(np.nan)
            acc_score.append(np.nan)
            if dataset_type == "dex-ycb":
                s_acc_list.append(np.nan)
                d_acc_list.append(np.nan)
                s_comp_list.append(np.nan)
                d_comp_list.append(np.nan)

        # ── Camera metrics ───────────────────────────────────────────────
        if "est_poses" in data and data["est_poses"] is not None and data["est_poses"].ndim >= 3:
            e_p = data["est_poses"]
            g_p = np.array([np.linalg.inv(rt) for rt in rts])
            e_i = data["est_intrinsics"]
            cam_mets = compute_camera_metrics(e_p, g_p, e_i, ks, s_val, R_val, tr_val)
            if not np.isnan(cam_mets["ate"]):
                ate_list.append(cam_mets["ate"])
                rpe_list.append(cam_mets["rpe"])
                rot_err_list.append(cam_mets["rot_error"])
                focal_err_list.append(cam_mets["focal_error"])
                pp_err_list.append(cam_mets["pp_error"])

        # ── Jitter data collection ───────────────────────────────────────
        if "pointmaps" in data:
            pm_j = normalize_array(data["pointmaps"], V, H, W)
            m_norm = normalize_array(m_2d, V, H, W, is_mask=True) if m_2d is not None else np.ones((V, H, W),
                                                                                                   dtype=bool)
            aligned_pm = np.empty_like(pm_j)
            for vi in range(V):
                aligned_pm[vi] = apply_similarity_transform(
                    pm_j[vi].reshape(-1, 3), s_val, R_val, tr_val
                ).reshape(H, W, 3)
            all_pointmaps_mv.append(aligned_pm)
            all_masks_mv.append(m_norm.astype(bool))

    # ── Aggregate metrics ────────────────────────────────────────────────
    chamfer_mean = float(np.nanmean(cham_dist))

    metrics = {
        "strategy": strategy_label,
        "n_frames": len(files),
        "chamfer": chamfer_mean,
        "completeness": float(np.nanmean(comp_score)),
        "accuracy": float(np.nanmean(acc_score)),
    }

    if dataset_type == "dex-ycb" and s_acc_list:
        m_static = float(np.nanmean(s_acc_list))
        m_dyn = float(np.nanmean(d_acc_list))
        metrics.update({
            "static_comp": float(np.nanmean(s_comp_list)),
            "dyn_comp": float(np.nanmean(d_comp_list)),
            "static_acc": m_static,
            "dyn_acc": m_dyn,
            "motion_gap": (m_static - m_dyn) if not (np.isnan(m_static) or np.isnan(m_dyn)) else np.nan,
        })

    # ── Camera metrics ───────────────────────────────────────────────────
    if ate_list:
        metrics.update({
            "ate": float(np.nanmean(ate_list)),
            "rpe": float(np.nanmean(rpe_list)),
            "rot_error": float(np.nanmean(rot_err_list)),
            "focal_error": float(np.nanmean(focal_err_list)),
            "pp_error": float(np.nanmean(pp_err_list)),
        })

    # ── Jitter ───────────────────────────────────────────────────────────
    if len(all_pointmaps_mv) >= 2:
        jitter = compute_static_jitter(all_pointmaps_mv, all_masks_mv, n_anchors=5000)
        if jitter:
            metrics.update(jitter)

    # ── Timing ───────────────────────────────────────────────────────────
    timing_path = os.path.join(os.path.dirname(files[0]), "timing.json")
    if os.path.exists(timing_path):
        try:
            with open(timing_path, "r", encoding="utf-8") as f_t:
                timing = json.load(f_t)
            metrics["align_frames"] = int(timing.get("n_frames", len(files)))
        except Exception:
            pass

    # ── Per-frame Chamfer plot ───────────────────────────────────────────
    if out_plot_dir:
        os.makedirs(out_plot_dir, exist_ok=True)
        frames = np.arange(len(files))
        plt.figure()
        plt.plot(frames, cham_dist, "g-s", label="Chamfer (per-frame)")
        plt.title(f"Chamfer Distance — {strategy_label}")
        plt.xlabel("Frame")
        plt.ylabel("Chamfer")
        plt.legend()
        plt.savefig(os.path.join(out_plot_dir, f"chamfer_{strategy_label}.png"))
        plt.close()

    return metrics, cham_dist


# ═══════════════════════════════════════════════════════════════════════════════
# Global Umeyama: solve one (s, R, t) from ALL frames × ALL views
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_multi_strategy(baseline_dir, dataset_root, view_label, subject_full, no_rerun=False,
                            dataset_type="dex-ycb"):
    """
    Evaluate the model output for:
      1. baseline_per_frame — Computes an independent Umeyama registration per frame to GT.
      2. global_4d — Evaluates the single holistic output (aligned_pts inherently produced with a single global Umeyama).
    """
    files = sorted(glob.glob(os.path.join(baseline_dir, "frame_*.npz")))
    if not files:
        print(f"  [WARN] No frames in {baseline_dir}")
        return []

    print(f"\\n  ── Evaluating {view_label} ({len(files)} frames) ──")

    plot_root = os.path.join("plots", dataset_type, subject_full, view_label)
    all_rows = []
    all_chs = []

    if not no_rerun:
        from vggt.utils.rerun_logging import initialize_rerun_session
        log_root = f"4d_eval_{view_label}"
        initialize_rerun_session(f"4d_eval_{subject_full}_{view_label}", RERUN_ADDR, log_root)

    # ── Tier 1: Global 4D Output ───────────────────────────
    print(f"  [TIER 1] Global 4D Alignment (Native 4D consistency) ...")
    global_label = f"global_4d_{view_label}"
    # By default _compute_metrics_for_alignment uses data["aligned_pts"] which from run_all_at_once_pipeline is the 4D global alignment.
    global_row, global_ch = _compute_metrics_for_alignment(
        files, global_label, dataset_root,
        global_transform=None,
        out_plot_dir=os.path.join(plot_root, "global_4d"),
        dataset_type=dataset_type,
    )
    if global_row is not None:
        global_row["subject"] = subject_full
        all_rows.append(global_row)
        all_chs.append((global_label, global_ch))
        chamfer_4d = global_row["chamfer"]
    else:
        chamfer_4d = float("nan")

    # ── Tier 2: Per-frame Baseline ───────────────────────────
    print(f"  [TIER 2] Per-frame Baseline (Independent Umeyama per frame) ...")
    baseline_label = f"baseline_per_frame_{view_label}"

    # We will simulate per-frame transforms by generating per-frame alignments
    from vggt.utils.umeyama_alignment import estimate_similarity_transform
    from vggt.utils.alignment_4d import normalize_spatial_dims, normalize_array

    first_data = np.load(files[0], allow_pickle=True)
    V, H, W = normalize_spatial_dims(first_data)

    per_frame_T = []
    for i, file in enumerate(files):
        data = np.load(file, allow_pickle=True)
        pm = normalize_array(data["pointmaps"], V, H, W)
        conf = normalize_array(data["pointmaps_confs"], V, H, W)
        m_2d = data.get("masks_2d")
        gt_pts = data["gt_pts"]

        t_idx = int(data["frame_idx"])
        ks = data["Ks"]
        view_names = [discover_view_name(dataset_root, k, dataset_type=dataset_type) for k in ks]

        # ── Global threshold for the whole frame ──
        frame_thr = np.quantile(conf, 1.0 - CONF_PERCENTILE) if conf is not None else 0.0

        vmasks = build_gt_validity_masks(
            t_idx, view_names, dataset_root,
            depth_max_m=DEPTH_MAX_M, target_hw=(H, W),
            dataset_type=dataset_type,
        )

        all_src, all_dst = [], []
        # Gather correspondences just like in get_single_view_correspondences, but we do it manually or call the tool
        for v in range(V):
            pts_flat = pm[v].reshape(-1, 3)
            conf_flat = conf[v].ravel()
            static = m_2d[v].ravel() if (m_2d is not None and m_2d.ndim == 3) else (
                m_2d.ravel() if m_2d is not None else None)
            src, dst = get_single_view_correspondences(
                t_idx, view_names[v], pm[v], conf[v], dataset_root,
                static_mask=static, conf_percentile=CONF_PERCENTILE,
                use_static_mask=False, dataset_type=dataset_type,
            )
            if src is not None and len(src) > 0:
                all_src.append(src)
                all_dst.append(dst)

        if all_src:
            s_cat = np.concatenate(all_src)
            d_cat = np.concatenate(all_dst)
            s_i, R_i, tr_i = estimate_similarity_transform(s_cat, d_cat)
        else:
            s_i, R_i, tr_i = 1.0, np.eye(3), np.zeros(3)

        per_frame_T.append((s_i, R_i, tr_i))

    # To use _compute_metrics_for_alignment with per_frame but NO GT global registration since we already mapped to GT world
    # we just pass a global GT reg of identity
    identity_gt = (1.0, np.eye(3), np.zeros(3))

    baseline_row, baseline_ch = _compute_metrics_for_alignment(
        files, baseline_label, dataset_root,
        per_frame_transforms=per_frame_T,
        gt_registration=identity_gt,
        out_plot_dir=os.path.join(plot_root, "baseline_per_frame"),
        dataset_type=dataset_type,
    )
    if baseline_row is not None:
        baseline_row["chamfer_4d"] = chamfer_4d
        baseline_row["delta_consistency"] = chamfer_4d - baseline_row["chamfer"]
        baseline_row["subject"] = subject_full
        all_rows.append(baseline_row)
        all_chs.append((baseline_label, baseline_ch))
        chamfer_3d = baseline_row["chamfer"]
    else:
        chamfer_3d = float("nan")

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\\n    ┌─ {view_label} {'─' * (50 - len(view_label))}")
    print(f"    │ Chamfer₃D (per-frame):  {chamfer_3d:.6f}")
    if global_row is not None:
        lbl = global_row.get("strategy", "")
        c3d = global_row.get("chamfer", float("nan"))
        dc = global_row.get("delta_consistency", float("nan")) if global_row == baseline_row else (c3d - chamfer_3d)
        if global_row != baseline_row: global_row["delta_consistency"] = dc
        print(f"    │ {lbl:<24}  Chamfer={c3d:.6f}  Δ={dc:+.6f}")
    print(f"    └{'─' * 52}")

    # ── Multi-strategy Chamfer plot ───────────────────────────────────────
    os.makedirs(plot_root, exist_ok=True)
    frames_ax = np.arange(len(files))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    colours = ["b", "g", "m", "c"]
    for (lbl, chs), col in zip(all_chs, colours):
        if chs:
            mu = float(np.nanmean(chs))
            plt.plot(frames_ax[:len(chs)], chs, f"{col}-o",
                     label=f"{lbl}  μ={mu:.5f}", alpha=0.8)
    plt.title(f"Chamfer Distance — {view_label}")
    plt.xlabel("Frame")
    plt.ylabel("Chamfer Distance")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_root, f"chamfer_all_{view_label}.png"), dpi=150)
    plt.close()

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / Main
# ═══════════════════════════════════════════════════════════════════════════════

def print_metrics_summary(results_df, label):
    """Prints a comparison table for baseline vs global."""
    print(f"\n{'=' * 80}")
    print(f"  Performance Summary: {label}")
    print(f"{'=' * 80}")
    pd.set_option("display.precision", 5)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", None)

    cols_to_show = [
        "strategy", "n_frames",
        "chamfer", "chamfer_4d", "delta_consistency",
        "completeness", "accuracy", "static_comp", "dyn_comp", "static_acc", "dyn_acc", "motion_gap",
        "ate", "rpe", "rot_error", "focal_error", "pp_error",
        "jitter_mean", "jitter_std", "jitter_p95", "jitter_max",
        "drift_mean", "hf_jitter",
    ]
    cols_to_show = [c for c in cols_to_show if c in results_df.columns]
    print(results_df[cols_to_show].to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="VGGT4D Native 4D Evaluation — Dual-Tier Alignment"
    )
    parser.add_argument("--data", type=str, choices=["dex-ycb", "hi4d"], default="dex-ycb", help="Dataset to evaluate.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--subjects", nargs="+", type=str, help="Specific subject codes to evaluate.")
    parser.add_argument("--views", nargs="+", type=int,
                        help="View counts to evaluate (e.g. --views 2 3 4).")
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun logging.")

    # Legacy flags for DexYCB
    for code in SUBJECT_BY_CODE.keys():
        parser.add_argument(f"--{code}", action="store_true")

    args = parser.parse_args()
    dataset_type = args.data
    subj_map = get_subject_by_code(dataset_type)
    dataset_config = get_dataset_config(dataset_type)

    if args.all:
        subjects = list(subj_map.keys())
    elif args.subjects:
        subjects = args.subjects
    else:
        # Legacy flag check
        import sys
        subjects = [a.lstrip('-') for a in sys.argv if a.startswith('--') and a.lstrip('-') in subj_map]
        if not subjects:
            print(f"[WARN] No subject selection provided; defaulting to first subject.")
            subjects = [list(subj_map.keys())[0]]

    view_set = set(args.views) if args.views else None

    for scode in subjects:
        subject_full = subj_map.get(scode)
        if not subject_full:
            continue

        dataset_root = os.path.join(dataset_config["root"], subject_full)
        subject_results = []

        print(f"\n{'━' * 80}")
        print(f"  Subject: {subject_full}  Dataset: {dataset_type}")
        print(f"{'━' * 80}")

        subject_dir = os.path.join("aligned_outputs", "baseline", dataset_type, subject_full)
        if not os.path.isdir(subject_dir):
            print(f"[WARN] No baseline outputs for {subject_full} in {subject_dir}")
            continue

        # Find available view directories
        view_dirs = sorted([
            d for d in os.listdir(subject_dir)
            if re.match(r"^\d+views$", d) and os.path.isdir(os.path.join(subject_dir, d))
        ])
        # Filter by requested views
        if view_set:
            view_dirs = [d for d in view_dirs if int(d.split("views")[0]) in view_set]

        if not view_dirs:
            print(f"[WARN] No matching view directories for {subject_full}")
            continue

        for view_dir in view_dirs:
            baseline_dir = os.path.join(subject_dir, view_dir)
            rows = evaluate_multi_strategy(
                baseline_dir, dataset_root, view_dir, subject_full,
                no_rerun=args.no_rerun, dataset_type=dataset_type
            )
            subject_results.extend(rows)

        if subject_results:
            df = pd.DataFrame(subject_results)
            print_metrics_summary(df, subject_full)
            safe_code = scode.replace("/", "_")
            out_csv = f"eval_summary_{dataset_type}_{safe_code}.csv"
            df.to_csv(out_csv, index=False)
            print(f"[INFO] Saved report to {out_csv}")


if __name__ == "__main__":
    main()
