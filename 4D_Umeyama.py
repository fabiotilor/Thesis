import os
import glob
import time
import argparse
import json
import re
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
    strategy3_pgo,
    solve_final_gt_registration,
    compute_4d_jitter_complete,
    normalize_spatial_dims,
    normalize_array
)

from eval_config import (
    DATASET_BASE_ROOT, SUBJECT_NAMES, SUBJECT_BY_CODE,
    MIN_CONF_THR, RERUN_ADDR, RERUN_EYE_UP
)
# ── camera discovery ───────────────────────────────────────────────────────────
# Moved to utils.camera_utils

# ── rerun configuration ────────────────────────────────────────────────────────

from mast3r.utils.rerun_logging import (
    configure_rerun_view_defaults,
    log_cameras_rerun,
    log_gt_sequence,
    log_aligned_sequence,
    log_pointcloud
)


def initialize_rerun_session(app_id, rerun_addr, log_root):
    """
    Initialize rerun with robust fallback:
    1) try external rerun server (grpc),
    2) if unavailable, spawn local viewer.
    """
    try:
        rr.init(app_id, spawn=False)
        rr.connect_grpc(rerun_addr)
        print(f"  [RERUN] Connected to existing viewer at {rerun_addr}")
    except Exception as e:
        print(f"  [RERUN][WARN] Could not connect to {rerun_addr}: {e}")
        print("  [RERUN] Spawning local viewer instead...")
        rr.init(app_id, spawn=True)

    rr.log(log_root, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    configure_rerun_view_defaults(log_root, RERUN_EYE_UP)


def save_aligned_results(
    frame_paths,
    frame_transforms,
    s_glob,
    R_glob,
    tr_glob,
    subject_name,
    strategy_label=None,
    dataset_root=None,
    out_dir=None,
    method_label=None,
    skip_existing_frames=True,
):
    """Saves evaluation-ready .npz files for each frame (method-namespaced output optional)."""
    if out_dir is None:
        if strategy_label is None:
            raise ValueError("save_aligned_results: either out_dir or strategy_label must be provided.")
        out_dir = os.path.join("aligned_outputs", subject_name, strategy_label)

    os.makedirs(out_dir, exist_ok=True)
    print(f"  [SAVE] Exporting aligned frames to {out_dir}...")

    for i, path in enumerate(frame_paths):
        try:
            data = np.load(path)
            V, H, W = normalize_spatial_dims(data)
            if H == 0:
                continue

            # 1. Standardize GT Points to Meters
            gt_pts = data["gt_pts"]
            if np.any(np.linalg.norm(gt_pts, axis=-1) > 10.0):
                gt_pts = gt_pts / 1000.0

            # 2. Re-calculate Aligned Full Pointcloud
            pm = normalize_array(data["pointmaps"], V, H, W).astype(np.float32)
            s_i, R_i, tr_i = frame_transforms[i]
            s_tot = s_glob * s_i
            R_tot = R_glob @ R_i
            tr_tot = s_glob * (R_glob @ tr_i) + tr_glob

            # Merge views for the "aligned_pts" field
            all_pts = []
            conf = (
                normalize_array(data["pointmaps_confs"], V, H, W)
                if "pointmaps_confs" in data
                else None
            )
            t, ks = int(data["frame_idx"]), data["Ks"]
            view_names = [discover_view_name(dataset_root, k) for k in ks]
            vmasks = build_gt_validity_masks(t, view_names, dataset_root, target_hw=(H, W))

            for v in range(V):
                mask = np.ones((H, W), dtype=bool)
                if vmasks[v] is not None:
                    mask &= vmasks[v]
                elif view_names[v] is not None:
                    print(f"    [WARN] Frame {t} view {v} ({view_names[v]}): Depth mask missing.")
                else:
                    print(f"    [WARN] Frame {t} view {v}: Could not discover view name for K.")

                if conf is not None:
                    mask &= conf[v] > MIN_CONF_THR

                p_v = pm[v][mask]
                if len(p_v) > 0:
                    all_pts.append(apply_similarity_transform(p_v, s_tot, R_tot, tr_tot))

            n_pts = sum(len(p) for p in all_pts)
            if n_pts == 0:
                print(
                    f"    [ERROR] Frame {t}: No points survived filtering "
                    f"(Conf > {MIN_CONF_THR} + GT Masks)."
                )

            aligned_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))

            # 3. Save bundled NPZ
            if skip_existing_frames:
                out_path = os.path.join(out_dir, f"frame_{t:04d}.npz")
                if os.path.exists(out_path):
                    print(f"  [SKIP] Existing {os.path.basename(out_path)} found in {out_dir}")
                    continue

            save_dict = {
                "gt_pts": gt_pts,
                "aligned_pts": aligned_pts,
                "frame_idx": int(t),
                "Ks": ks,
                "R_ts": data["R_ts"],
                "masks_2d": data["masks_2d"],
                "est_poses": data.get("est_poses"),
                "est_intrinsics": data.get("est_intrinsics"),
                "pointmaps": data["pointmaps"],
                "pointmaps_confs": data.get("pointmaps_confs"),
                "scale": float(s_tot),
                "R": R_tot,
                "tr": tr_tot,
            }

            out_path = os.path.join(out_dir, f"frame_{t:04d}.npz")
            np.savez_compressed(out_path, **save_dict)
        except Exception as e:
            print(f"    [ERROR] Strategy save failed for frame={i} ({path}): {e}")
            continue


def save_timing(out_dir, strategy_label, n_frames, total_seconds):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "strategy": strategy_label,
        "n_frames": int(n_frames),
        "total_seconds": float(total_seconds),
        "seconds_per_frame": float(total_seconds / max(n_frames, 1)),
    }
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  [TIME] {strategy_label}: total={payload['total_seconds']:.2f}s  per_frame={payload['seconds_per_frame']:.3f}s")


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate_subject(subject_name, selected_views=None, pgo_only=False):
    base_dir = os.path.join("aligned_outputs", subject_name)
    if not os.path.exists(base_dir): return

    view_dir_pattern = re.compile(r"^\d+views$")
    view_dirs = sorted(
        [d for d in os.listdir(base_dir) if view_dir_pattern.match(d) and os.path.isdir(os.path.join(base_dir, d))]
    )
    if not view_dirs: return

    if selected_views:
        selected_view_dirs = {f"{v}views" for v in selected_views}
        view_dirs = [vdir for vdir in view_dirs if vdir in selected_view_dirs]
        if not view_dirs:
            print(f"[WARN] No matching view folders found for {subject_name} and views {selected_views}")
            return

    for vdir in view_dirs:
        in_dir = os.path.join(base_dir, vdir)
        paths = sorted(glob.glob(os.path.join(in_dir, "frame_*.npz")),
                       key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        if len(paths) < 2: continue

        dataset_root = os.path.join(DATASET_BASE_ROOT, subject_name)
        log_root = f"4d_eval_{vdir}"
        initialize_rerun_session(f"4d_eval_{subject_name}_{vdir}", RERUN_ADDR, log_root)

        # 0. GT
        print(f"  [RERUN] Logging GT for {vdir}...")
        log_gt_sequence(paths, log_root=log_root)

        if not pgo_only:
            # 1. Strategy 1 (Reference Frame 0)
            print(f"\n--- [Strategy 1] Reference Frame Alignment ({vdir}) ---")
            start_s1 = time.perf_counter()
            tf_s1 = strategy1_reference(paths, dataset_root)
            s_g1, R_g1, tr_g1 = solve_final_gt_registration(paths, tf_s1, dataset_root)

            strat1_label = f"Strategy_1_{vdir}"
            save_aligned_results(paths, tf_s1, s_g1, R_g1, tr_g1, subject_name, strat1_label, dataset_root)
            log_aligned_sequence(paths, tf_s1, s_g1, R_g1, tr_g1, "Strategy_1", [0, 0, 255], dataset_root,
                                 log_root=log_root)
            save_timing(
                os.path.join("aligned_outputs", subject_name, strat1_label),
                strat1_label,
                len(paths),
                time.perf_counter() - start_s1,
            )

            # 2. Strategy 2 (Hierarchical)
            print(f"\n--- [Strategy 2] Hierarchical Alignment ({vdir}) ---")
            start_s2 = time.perf_counter()
            tf_s2 = strategy2_hierarchical(paths, dataset_root)
            s_g2, R_g2, tr_g2 = solve_final_gt_registration(paths, tf_s2, dataset_root)

            strat2_label = f"Strategy_2_{vdir}"
            save_aligned_results(paths, tf_s2, s_g2, R_g2, tr_g2, subject_name, strat2_label, dataset_root)
            log_aligned_sequence(paths, tf_s2, s_g2, R_g2, tr_g2, "Strategy_2", [255, 0, 0], dataset_root,
                                 log_root=log_root)
            save_timing(
                os.path.join("aligned_outputs", subject_name, strat2_label),
                strat2_label,
                len(paths),
                time.perf_counter() - start_s2,
            )

        # 3. Strategy 3 (PGO)
        print(f"\n--- [Strategy 3] Pose Graph Optimization ({vdir}) ---")
        start_s3 = time.perf_counter()
        tf_s3 = strategy3_pgo(paths, dataset_root, num_iters=50)
        s_g3, R_g3, tr_g3 = solve_final_gt_registration(paths, tf_s3, dataset_root)

        strat3_label = f"Strategy_3_{vdir}"
        save_aligned_results(paths, tf_s3, s_g3, R_g3, tr_g3, subject_name, strat3_label, dataset_root)
        log_aligned_sequence(paths, tf_s3, s_g3, R_g3, tr_g3, "Strategy_3", [0, 255, 0], dataset_root,
                             log_root=log_root)
        save_timing(
            os.path.join("aligned_outputs", subject_name, strat3_label),
            strat3_label,
            len(paths),
            time.perf_counter() - start_s3,
        )

        print(f"\n[INFO] Aligned results saved for {subject_name} ({vdir}). Run evaluate_4D.py to see metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pgo", action="store_true", help="Run only Strategy 3 (Pose Graph Optimization).")
    parser.add_argument("--views", nargs="+", type=int, help="Optional view counts to process (e.g. --views 2 3 4).")
    for code in SUBJECT_BY_CODE.keys(): parser.add_argument(f"--{code}", action="store_true")
    args = parser.parse_args()

    selected = [v for k, v in SUBJECT_BY_CODE.items() if getattr(args, k)]
    if args.all: selected = SUBJECT_NAMES
    for s in selected:
        evaluate_subject(s, selected_views=args.views, pgo_only=args.pgo)
