"""Summarize per-round timing records produced by the FedAVGTrainer timing patch.

Usage:
    python summarize_efficiency.py /path/to/one/experiment_directory
"""

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python summarize_efficiency.py EXPERIMENT_DIRECTORY")

    experiment = Path(sys.argv[1])
    timing_files = sorted(experiment.glob("*_timing.csv"))
    if not timing_files:
        raise SystemExit(f"No *_timing.csv files found in {experiment}")

    records = pd.concat((pd.read_csv(path) for path in timing_files), ignore_index=True)
    average = records.groupby("stage", as_index=False)["seconds"].mean()
    train = average.loc[average["stage"] == "local_train", "seconds"]
    if train.empty:
        raise SystemExit("The timing files do not contain local_train records.")

    train_seconds = float(train.iloc[0])
    output_rows = []
    for stage in ("snapshot_eval_global", "snapshot_eval_local"):
        matched = average.loc[average["stage"] == stage, "seconds"]
        if not matched.empty:
            eval_seconds = float(matched.iloc[0])
            output_rows.append(
                {
                    "model_snapshot": stage.removeprefix("snapshot_eval_"),
                    "mean_local_train_seconds_per_round": train_seconds,
                    "mean_snapshot_evaluation_seconds_per_round": eval_seconds,
                    "collection_overhead_ratio": eval_seconds / train_seconds,
                    "note": "per-round wall-clock overhead of collecting the per-sample trajectory; this is not the complete audit time",
                }
            )

    summary = pd.DataFrame(output_rows)
    output = experiment / "trajectory_collection_timing_summary.csv"
    summary.to_csv(output, index=False, float_format="%.4f")
    print("Per-round trajectory-collection overhead")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {output}")

    audit_path = experiment / "audit_timing.csv"
    if not audit_path.exists():
        print("\nNo audit_timing.csv yet. Run 3_run_audit.py after applying its timing patch.")
        return

    audit = pd.read_csv(audit_path)
    audit_summary = (
        audit.groupby(
            ["target_model_type", "target_round", "stage", "alg"], dropna=False, as_index=False
        )["seconds"]
        .mean()
        .rename(columns={"seconds": "mean_wall_seconds"})
    )
    audit_output = experiment / "posthoc_audit_timing_summary.csv"
    audit_summary.to_csv(audit_output, index=False, float_format="%.4f")
    print("\nComplete post-hoc audit timing (trajectory load + each algorithm's slope/ROC/TPR work)")
    print(audit_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {audit_output}")

    # For each audit method, compare its complete post-hoc cost (trajectory
    # preparation plus score/slope/ROC/TPR computation) with the accumulated
    # local-training time of the same client through the target round.
    keys = [
        "target_model_type",
        "target_round",
        "start_round",
        "random_seed",
        "client_idx",
    ]
    preparation = (
        audit[audit["stage"] == "load_and_prepare_trajectory"]
        .groupby(keys, as_index=False)["seconds"]
        .mean()
        .rename(columns={"seconds": "prepare_seconds"})
    )
    algorithms = audit[audit["stage"] == "algorithm_and_metrics"].copy()
    rows = []
    for _, audit_row in algorithms.iterrows():
        train_rows = records[
            (records["client_idx"] == audit_row["client_idx"])
            & (records["stage"] == "local_train")
            & (records["round"] >= audit_row["start_round"])
            & (records["round"] <= audit_row["target_round"])
        ]
        matching_prepare = preparation.copy()
        for key in keys:
            matching_prepare = matching_prepare[matching_prepare[key] == audit_row[key]]
        if train_rows.empty or matching_prepare.empty:
            continue
        training_seconds = float(train_rows["seconds"].sum())
        full_audit_seconds = float(matching_prepare["prepare_seconds"].iloc[0]) + float(
            audit_row["seconds"]
        )
        rows.append(
            {
                "model_snapshot": audit_row["target_model_type"],
                "target_round": audit_row["target_round"],
                "client_idx": audit_row["client_idx"],
                "method": audit_row["alg"],
                "training_seconds_through_target_round": training_seconds,
                "full_posthoc_audit_seconds": full_audit_seconds,
                "posthoc_audit_to_training_ratio": full_audit_seconds / training_seconds,
                "note": "post-hoc implementation overhead, not the paper's unpublished per-round GPU-time measurement",
            }
        )

    if rows:
        efficiency = pd.DataFrame(rows)
        efficiency_output = experiment / "efficiency_summary.csv"
        efficiency.to_csv(efficiency_output, index=False, float_format="%.4f")
        print("\nComplete-audit / accumulated-training ratios")
        print(efficiency.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"\nSaved: {efficiency_output}")


if __name__ == "__main__":
    main()
