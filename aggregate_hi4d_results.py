#!/usr/bin/env python3
import os
import glob
import pandas as pd


def main():
    # Find all individual hi4d subject summaries
    csv_files = sorted(glob.glob("eval_summary_hi4d_pair*.csv"))

    if not csv_files:
        print("[ERROR] No individual hi4d subject summary CSV files found in the current directory.")
        return

    print(f"[INFO] Found {len(csv_files)} subject summaries to aggregate:")
    for f in csv_files:
        print(f"  - {f}")

    # Load and concat all dataframes
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Average numeric columns grouped by strategy
    numeric_df = combined_df.select_dtypes(include="number")
    numeric_df["strategy"] = combined_df["strategy"]

    aggregated = (
        numeric_df.groupby("strategy").mean().reset_index().sort_values(by="strategy")
    )

    # Format and print
    print("\n" + "=" * 100)
    print("CROSS-SUBJECT AGGREGATED RESULTS - HI4D (MEAN ACROSS PROCESSED SUBJECTS)")
    print("=" * 100)

    pd.set_option("display.precision", 5)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", None)

    cols_to_show = [
        "strategy", "n_frames", "chamfer", "chamfer_4d", "delta_consistency",
        "completeness", "accuracy",
        "ate", "rpe", "rot_error", "focal_error", "pp_error",
        "jitter_mean", "jitter_std", "jitter_p95", "jitter_max",
        "drift_mean", "hf_jitter"
    ]

    cols_to_show = [c for c in cols_to_show if c in aggregated.columns]
    print(aggregated[cols_to_show].to_string(index=False))
    print("=" * 100)

    # Save the aggregated results
    out_file = "eval_summary_hi4d_ALL_SUBJECTS.csv"
    aggregated.to_csv(out_file, index=False)
    print(f"[INFO] Aggregated results saved to {out_file}\n")


if __name__ == "__main__":
    main()
