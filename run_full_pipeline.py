#!/usr/bin/env python3
import sys
import subprocess
import glob
import pandas as pd


def run_command(cmd):
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        sys.exit(e.returncode)


def main():
    args = sys.argv[1:]
    pgo_only = "--pgo" in args

    # Ensure that Python executes the scripts properly in the environment
    python_exec = "python"

    if pgo_only:
        # Fast path: reuse existing aligned baseline outputs and evaluate only PGO for subject 01 on 2/3/4 views.
        scoped_args = ["--01", "--views", "2", "3", "4", "--pgo"]
        run_command([python_exec, "4D_Umeyama.py"] + scoped_args)
        run_command([python_exec, "evaluate_4D.py"] + scoped_args)
    else:
        # 1. Run Baseline Alignment (runs MASt3R + initial Umeyama per-frame alignments)
        run_command([python_exec, "align_reconstruction_umeyama.py"] + args)

        # 2. Run 4D Strategies
        run_command([python_exec, "4D_Umeyama.py"] + args)

        # 3. Evaluate Framework
        run_command([python_exec, "evaluate_4D.py"] + args)

    # 4. Aggregate Results Across Scripts
    csv_files = glob.glob("eval_summary_*.csv")
    csv_files = [f for f in csv_files if "ALL_SUBJECTS" not in f]

    if not csv_files:
        print("[WARN] No CSV files found to aggregate.")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    if combined_df.empty:
        print("[WARN] Combined CSV is empty.")
        return

    # Select numeric columns for the mean operation
    numeric_df = combined_df.select_dtypes(include='number')
    numeric_df['strategy'] = combined_df['strategy']

    # Group by strategy (e.g. '4views', 'Strategy_1_4views')
    aggregated = numeric_df.groupby('strategy').mean().reset_index()

    # Optional sorting so baseline precedes Strategy_1 precedes Strategy_2
    aggregated = aggregated.sort_values(by='strategy')

    print("\n\n" + "=" * 80)
    print("CROSS-SUBJECT AGGREGATED RESULTS (MEAN ACROSS ALL PROCESSED SUBJECTS)")
    print("=" * 80)

    pd.set_option('display.precision', 5)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_columns', None)

    print(aggregated.to_string(index=False))

    out_file = "eval_summary_ALL_SUBJECTS.csv"
    aggregated.to_csv(out_file, index=False)
    print(f"\n[INFO] Aggregated results saved to {out_file}")


if __name__ == "__main__":
    main()
