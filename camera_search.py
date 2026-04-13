#!/usr/bin/env python3
import itertools
import json
import os
import shutil
import time
import traceback
from typing import Dict, List, Tuple

import numpy as np
import torch

import align_reconstruction_umeyama as aru
from align_reconstruction_umeyama import MODEL_NAME, DEVICE
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.temporal_metrics import (
    compute_chamfer_distance,
    compute_accuracy,
    compute_completeness,
    split_points_by_mask,
)


DATASET_ROOT = "/home/fabio/datasets/dex-ycb-multiview/20200709-subject-01__20200709_141754"
ALL_CAMERAS = ["00", "01", "02", "03", "04", "05", "06", "07"]
VIEW_COUNTS = [2, 3, 4]
TAU = 0.01

RESULTS_DIR = "results"
RESULTS_JSONL = os.path.join(RESULTS_DIR, "camera_search_results.jsonl")
ERRORS_LOG = os.path.join(RESULTS_DIR, "camera_search_errors.log")
REPORT_TXT = os.path.join(RESULTS_DIR, "camera_search_report.txt")
TMP_RECON_ROOT = os.path.join(RESULTS_DIR, "_camera_search_tmp")

_MODEL = None


def combo_key(combo: Tuple[str, ...]) -> str:
    return ",".join(combo)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values: List[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.std(vals)) if vals else float("nan")


def load_done_combos() -> set:
    done = set()
    if not os.path.exists(RESULTS_JSONL):
        return done
    with open(RESULTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                c = obj.get("combo", [])
                if c:
                    done.add(combo_key(tuple(c)))
            except json.JSONDecodeError:
                continue
    return done


def append_jsonl(obj: Dict) -> None:
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def log_error(combo: Tuple[str, ...], exc: Exception) -> None:
    with open(ERRORS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} combo={combo} error={repr(exc)}\n")
        f.write(traceback.format_exc() + "\n")


def get_frame_iter(frames):
    try:
        from tqdm import tqdm
        return tqdm(frames, leave=False)
    except Exception:
        return frames


def get_or_create_model():
    global _MODEL
    if _MODEL is None:
        print(f"[INFO] Loading model '{MODEL_NAME}' once for search...")
        _MODEL = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    return _MODEL


def call_run_reconstruction(combo: Tuple[str, ...], dataset_root: str):
    """
    Calls the current local run_reconstruction implementation and returns
    frame dicts by reading generated frame_*.npz files.
    """
    model = get_or_create_model()
    run_name = f"k{len(combo)}_{combo_key(combo).replace(',', '_')}"
    out_dir = os.path.join(TMP_RECON_ROOT, run_name)
    cache_root = os.path.join(TMP_RECON_ROOT, "cache")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    try:
        aru.run_reconstruction(
            model=model,
            dataset_root=dataset_root,
            target_views=list(combo),
            out_dir=out_dir,
            cache_root=cache_root,
            flow_threshold=1.0,
            run_tag=f"camera_search_{run_name}",
        )

        files = sorted(
            f for f in os.listdir(out_dir)
            if f.startswith("frame_") and f.endswith(".npz")
        )
        if not files:
            raise RuntimeError(f"No frame outputs produced in {out_dir}")

        frame_results = []
        for fname in files:
            path = os.path.join(out_dir, fname)
            with np.load(path, allow_pickle=False) as data:
                frame = {k: data[k] for k in data.files}
            frame_results.append(frame)
        return frame_results
    finally:
        # Keep disk usage bounded during long searches.
        shutil.rmtree(out_dir, ignore_errors=True)


def evaluate_combo(combo: Tuple[str, ...]) -> Dict:
    frame_results = call_run_reconstruction(combo, DATASET_ROOT)
    if not isinstance(frame_results, list) or not frame_results:
        raise RuntimeError("run_reconstruction returned no per-frame results.")

    chamfer_static_vals = []
    acc_static_vals = []
    acc_dynamic_vals = []
    comp_static_vals = []

    for frame in get_frame_iter(frame_results):
        aligned_pts = frame["aligned_pts"]
        gt_pts = frame["gt_pts"]
        if "masks_2d" not in frame:
            continue
        masks_2d = frame["masks_2d"]
        Ks = frame["Ks"]
        R_ts = frame["R_ts"]

        static_pts, dynamic_pts = split_points_by_mask(aligned_pts, masks_2d, Ks, R_ts)
        gt_static_pts, gt_dynamic_pts = split_points_by_mask(gt_pts, masks_2d, Ks, R_ts)

        if len(static_pts) == 0 or len(gt_static_pts) == 0:
            chamfer_static = float("nan")
            acc_static = float("nan")
            comp_static = float("nan")
        else:
            chamfer_static = float(compute_chamfer_distance(static_pts, gt_static_pts))
            acc_static = float(compute_accuracy(static_pts, gt_static_pts, tau=TAU))
            comp_static = float(compute_completeness(static_pts, gt_static_pts, tau=TAU))

        if len(dynamic_pts) == 0 or len(gt_dynamic_pts) == 0:
            acc_dynamic = float("nan")
        else:
            acc_dynamic = float(compute_accuracy(dynamic_pts, gt_dynamic_pts, tau=TAU))

        chamfer_static_vals.append(chamfer_static)
        acc_static_vals.append(acc_static)
        acc_dynamic_vals.append(acc_dynamic)
        comp_static_vals.append(comp_static)

    mean_chamfer_static = safe_mean(chamfer_static_vals)
    mean_acc_static = safe_mean(acc_static_vals)
    mean_acc_dynamic = safe_mean(acc_dynamic_vals)
    mean_comp_static = safe_mean(comp_static_vals)
    std_acc_static = safe_std(acc_static_vals)
    std_acc_dynamic = safe_std(acc_dynamic_vals)

    return {
        "combo": list(combo),
        "n_views": len(combo),
        "mean_chamfer_static": mean_chamfer_static,
        "mean_acc_static": mean_acc_static,
        "mean_acc_dynamic": mean_acc_dynamic,
        "mean_comp_static": mean_comp_static,
        "std_acc_static": std_acc_static,
        "std_acc_dynamic": std_acc_dynamic,
        "motion_gap": mean_acc_static - mean_acc_dynamic if not (np.isnan(mean_acc_static) or np.isnan(mean_acc_dynamic)) else float("nan"),
    }


def rank_top(rows: List[Dict], key: str, ascending: bool, topk: int = 5) -> List[Dict]:
    valid = [r for r in rows if not np.isnan(r.get(key, np.nan))]
    return sorted(valid, key=lambda r: r[key], reverse=not ascending)[:topk]


def table_lines(rows: List[Dict], key: str, ascending: bool, title: str) -> List[str]:
    out = [title]
    ranked = rank_top(rows, key, ascending=ascending, topk=5)
    if not ranked:
        out.append("  (no valid entries)")
        return out
    for i, r in enumerate(ranked, 1):
        out.append(f"  {i:>2}. {tuple(r['combo'])} | {key}={r[key]:.6f}")
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TMP_RECON_ROOT, exist_ok=True)

    combos_by_k = {k: list(itertools.combinations(ALL_CAMERAS, k)) for k in VIEW_COUNTS}
    done = load_done_combos()

    total = sum(len(v) for v in combos_by_k.values())
    processed = 0
    started = time.time()

    for k in VIEW_COUNTS:
        combos = combos_by_k[k]
        n_k = len(combos)
        for idx, combo in enumerate(combos, 1):
            key = combo_key(combo)
            if key in done:
                processed += 1
                continue

            try:
                result = evaluate_combo(combo)
                append_jsonl(result)
                done.add(key)
                processed += 1
                chamfer_str = f"{result['mean_chamfer_static']:.4f}" if not np.isnan(result["mean_chamfer_static"]) else "nan"
            except Exception as exc:
                log_error(combo, exc)
                processed += 1
                chamfer_str = "ERR"

            elapsed = time.time() - started
            avg_per = elapsed / max(processed, 1)
            remaining = total - processed
            eta = avg_per * max(remaining, 0)
            print(
                f"[k={k} | {idx}/{n_k} | elapsed {format_seconds(elapsed)} | "
                f"ETA {format_seconds(eta)}] {combo} chamfer={chamfer_str}"
            )
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # Load all successful rows for final report.
    rows = []
    if os.path.exists(RESULTS_JSONL):
        with open(RESULTS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    rows_by_k = {k: [r for r in rows if r.get("n_views") == k] for k in VIEW_COUNTS}

    report = []
    report.append("Camera Search Report")
    report.append(f"Dataset: {DATASET_ROOT}")
    report.append(f"Total combinations: {total}")
    report.append("")

    best_by_chamfer = {}
    for k in VIEW_COUNTS:
        report.append(f"=== k={k} views ===")
        rk = rows_by_k[k]
        report.extend(table_lines(rk, "mean_chamfer_static", ascending=True, title="Top-5 by mean_chamfer_static"))
        report.extend(table_lines(rk, "mean_acc_static", ascending=False, title="Top-5 by mean_acc_static"))
        report.extend(table_lines(rk, "mean_acc_dynamic", ascending=False, title="Top-5 by mean_acc_dynamic"))
        report.extend(table_lines(rk, "motion_gap", ascending=True, title="Top-5 by motion_gap (lowest best)"))
        report.append("")

        top_ch = rank_top(rk, "mean_chamfer_static", ascending=True, topk=1)
        if top_ch:
            best_by_chamfer[k] = set(top_ch[0]["combo"])

    report.append("=== Nested Selection Check (best by chamfer) ===")
    for k in [2, 3]:
        s1 = best_by_chamfer.get(k)
        s2 = best_by_chamfer.get(k + 1)
        if s1 is None or s2 is None:
            report.append(f"k={k}->k={k+1}: unavailable")
            continue
        holds = s1.issubset(s2)
        report.append(
            f"k={k}->k={k+1}: {'HOLDS' if holds else 'DOES NOT HOLD'} | "
            f"best{k}={sorted(s1)} best{k+1}={sorted(s2)}"
        )

    report_text = "\n".join(report)
    print(report_text)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")


if __name__ == "__main__":
    main()
