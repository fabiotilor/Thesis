#!/usr/bin/env python3
"""
Aggregate eval_summary_dex-ycb_XX.csv files (subjects 01–10) into a
single mean / mean±std table.

Derived quantities (motion_gap, delta_consistency) are recomputed from
the aggregated means rather than averaged directly.

Usage:
    python aggregate_evals.py
    python aggregate_evals.py --pattern "eval_summary_dex-ycb_*.csv"
    python aggregate_evals.py --pattern "eval_summary_hi4d_*.csv"
"""

import argparse
import glob
import re

import numpy as np
import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_view_suffix(label: str) -> str | None:
    m = re.search(r"(\d+views)$", str(label))
    return m.group(1) if m else None


def is_baseline_label(label: str) -> bool:
    s = str(label)
    return bool(re.fullmatch(r"\d+views", s) or s.startswith("baseline_"))


def add_delta_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Δconsistency = chamfer(strategy) − chamfer(baseline), matched per view-count.
    Supports both:
      - new labels : baseline_2views, strategy1_2views, …
      - legacy labels: 2views, Strategy_1_2views, …
    Should be called on the *aggregated* mean DataFrame so the difference
    is computed from averaged chamfer values, not averaged deltas.
    """
    if df.empty or "strategy" not in df.columns or "chamfer" not in df.columns:
        return df

    df = df.copy()
    df["delta_consistency"] = np.nan

    baseline_chamfer: dict[str, float] = {}
    for _, row in df.iterrows():
        if is_baseline_label(row["strategy"]):
            vs = extract_view_suffix(row["strategy"])
            if vs is not None:
                baseline_chamfer[vs] = row["chamfer"]

    for idx, row in df.iterrows():
        if is_baseline_label(row["strategy"]):
            continue
        vs = extract_view_suffix(row["strategy"])
        if vs is None:
            continue
        base = baseline_chamfer.get(vs)
        if base is None or np.isnan(base):
            continue
        df.at[idx, "delta_consistency"] = row["chamfer"] - base

    return df


# ── Columns that must never be averaged directly ──────────────────────────────
# They are either metadata or derived from other averaged columns.
NON_AGG = {"strategy", "n_frames", "subject",
           "per_frame_jitter", "delta_consistency", "motion_gap"}

PRINT_ORDER = [
    "chamfer", "delta_consistency",
    "completeness", "static_comp", "dyn_comp",
    "static_acc", "dyn_acc", "motion_gap",
    "overall_acc", "overall_comp",
    "ate", "rpe", "rot_error", "focal_error", "pp_error",
    "jitter_mean", "jitter_std", "jitter_p95", "jitter_max",
    "drift_mean", "hf_jitter",
    "n_anchors", "align_frames",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def aggregate(pattern: str) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern!r}")
    print(f"Found {len(files)} file(s): {', '.join(files)}\n")

    # Load & concatenate
    frames = []
    for path in files:
        df = pd.read_csv(path, quotechar='"', skipinitialspace=True)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Drop columns that cannot or should not be aggregated
    drop = [c for c in combined.columns if c in NON_AGG - {"strategy", "n_frames"}]
    combined.drop(columns=drop, errors="ignore", inplace=True)

    # Coerce everything except strategy/n_frames to numeric
    numeric_cols = [c for c in combined.columns
                    if c not in ("strategy", "n_frames")]
    combined[numeric_cols] = combined[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Group by strategy — mean and std across subjects
    mean_df = (combined
               .groupby("strategy", sort=False)[numeric_cols]
               .mean()
               .reset_index())
    std_df  = (combined
               .groupby("strategy", sort=False)[numeric_cols]
               .std()
               .reset_index())

    # Attach n_frames (constant across subjects; take first)
    n_frames = (combined
                .groupby("strategy", sort=False)["n_frames"]
                .first()
                .reset_index())
    mean_df = n_frames.merge(mean_df, on="strategy")

    # ── Recompute derived quantities from aggregated means ────────────────
    mean_df["motion_gap"] = mean_df["static_acc"] - mean_df["dyn_acc"]
    mean_df = add_delta_consistency(mean_df)

    # std of derived quantities requires error propagation — leave as NaN
    std_df["motion_gap"]        = np.nan
    std_df["delta_consistency"] = np.nan

    # ── Print ─────────────────────────────────────────────────────────────
    strategies = mean_df["strategy"].tolist()
    col_w = 22
    val_w = 22

    print(f"Aggregated over {len(files)} subject(s)\n")
    header = f"{'metric':<{col_w}}" + "".join(f"{s:>{val_w}}" for s in strategies)
    print(header)
    print("─" * len(header))

    print_cols = [c for c in PRINT_ORDER if c in mean_df.columns]
    for col in print_cols:
        if mean_df[col].isna().all():
            continue    # skip all-NaN columns (e.g. overall_* on dex-ycb)

        row_str = f"  {col:<{col_w - 2}}"
        for _, mrow in mean_df.iterrows():
            srow = std_df[std_df["strategy"] == mrow["strategy"]].iloc[0]
            mu = mrow[col]
            sd = srow[col] if col in srow.index else np.nan

            if pd.isna(mu):
                cell = "—"
            elif pd.isna(sd) or col in ("motion_gap", "delta_consistency",
                                         "n_frames", "n_anchors", "align_frames"):
                cell = f"{mu:.5f}"
            else:
                cell = f"{mu:.5f} ± {sd:.5f}"
            row_str += f"{cell:>{val_w}}"
        print(row_str)

    # ── Save ──────────────────────────────────────────────────────────────
    # 1. Clean mean-only table
    mean_out = "eval_aggregate_mean.csv"
    mean_df.to_csv(mean_out, index=False, float_format="%.6f")

    # 2. Interleaved mean / std columns
    merged = mean_df.copy()
    for col in numeric_cols:
        if col in std_df.columns:
            std_vals = std_df.set_index("strategy")[col]
            merged[f"{col}_std"] = merged["strategy"].map(std_vals)
    merged["motion_gap_std"]        = np.nan
    merged["delta_consistency_std"] = np.nan
    merged_out = "eval_aggregate_mean_std.csv"
    merged.to_csv(merged_out, index=False, float_format="%.6f")

    print(f"\nSaved: {mean_out}")
    print(f"Saved: {merged_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="eval_summary_dex-ycb_*.csv",
        help="Glob pattern for input CSV files (default: eval_summary_dex-ycb_*.csv)",
    )
    args = parser.parse_args()
    aggregate(args.pattern)