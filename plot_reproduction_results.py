"""Create figures and tables from an already-finished 4-client FL experiment.

This script only reads the saved JSONL prediction trajectories and audit CSVs.
It does not train a model or change any saved experiment result.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "results" / "results" / "FedAvg" / "cifar10"
FIGURES = ROOT / "figures"
CLIENT_ID = 0

# Keep the order consistent in the two line charts and the two result tables.
METHODS = [
    "fed-loss",
    "fed-conf",
    "fed-rescaled",
    "delta-diff",
    "delta-ratio",
    "back-front-diff",
    "back-front-ratio",
    "01-loss",
    "our_loss",
    "our_conf",
    "our_rescaled",
]


def find_experiment() -> Path:
    matches = sorted(EXPERIMENTS.glob("*rounds-100"))
    if not matches:
        raise FileNotFoundError(
            "Cannot find the 100-round experiment directory under "
            f"{EXPERIMENTS}."
        )
    return matches[0]


def load_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = next(iter(obj))
            records[int(key)] = obj[key]
    return records


def slope(values: np.ndarray) -> np.ndarray:
    """OLS slope along axis 0; values has shape (rounds, samples)."""
    n_rounds = values.shape[0]
    if n_rounds < 2:
        return np.zeros(values.shape[1])
    x = np.arange(1, n_rounds + 1, dtype=float)
    x = x - x.mean()
    centered = values - values.mean(axis=0, keepdims=True)
    return (x[:, None] * centered).sum(axis=0) / (x**2).sum()


def find_split() -> dict:
    candidates = sorted((ROOT / "results" / "cifar10").glob("iidk_*_4_0.pkl"))
    if not candidates:
        raise FileNotFoundError("Cannot find results/cifar10/iidk_*_4_0.pkl")
    with candidates[0].open("rb") as handle:
        return pickle.load(handle)


def make_figure1(experiment: Path, split: dict) -> None:
    """Paper Figure 1-style local confidence/slope/histogram for client 0."""
    records = load_jsonl(experiment / f"{CLIENT_ID}_local_pred.jsonl")
    rounds = sorted(records)
    conf = np.asarray([records[r]["conf"] for r in rounds], dtype=float)

    n_train = len(split[CLIENT_ID]["train"])
    n_test = len(split[CLIENT_ID]["test"])
    member = conf[:, :n_train]
    nonmember = conf[:, n_train : n_train + n_test]

    member_mean = member.mean(axis=1)
    nonmember_mean = nonmember.mean(axis=1)

    # The middle panel uses the slope available at each current round: R1, R1-R2, ...
    # A slope needs at least two snapshots.  Do not invent a value for the
    # first round; plotting an artificial zero there creates a misleading
    # near-vertical segment at the far left of the middle panel.
    slope_rounds = np.asarray(rounds[1:]) + 1
    member_slopes = np.array([slope(member[: i + 1]).mean() for i in range(1, len(rounds))])
    nonmember_slopes = np.array(
        [slope(nonmember[: i + 1]).mean() for i in range(1, len(rounds))]
    )

    # R30 in the paper means use the first 30 stored snapshots, indexed 0--29 here.
    hist_count = min(30, len(rounds))
    member_hist = slope(member[:hist_count])
    nonmember_hist = slope(nonmember[:hist_count])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.asarray(rounds) + 1
    axes[0].plot(x, nonmember_mean, label="Non-member", color="#1f77b4")
    axes[0].plot(x, member_mean, label="Member", color="#e76f73")
    axes[0].set(title="Average confidence", xlabel="Communication round", ylabel="Confidence")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(slope_rounds, nonmember_slopes, label="Non-member", color="#1f77b4")
    axes[1].plot(slope_rounds, member_slopes, label="Member", color="#e76f73")
    axes[1].set(
        title="Slope available up to each round",
        xlabel="Communication round",
        ylabel="Slope of confidence",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    bins = np.histogram_bin_edges(np.r_[member_hist, nonmember_hist], bins=22)
    axes[2].hist(nonmember_hist, bins=bins, alpha=0.65, label="Non-member", color="#1f77b4")
    axes[2].hist(member_hist, bins=bins, alpha=0.65, label="Member", color="#e76f73")
    axes[2].set(
        title=f"Per-sample slopes at round {hist_count}",
        xlabel="Slope of confidence",
        ylabel="Number of samples",
    )
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.suptitle("Figure 1-style result: local model, CIFAR-10, client 0, seed 0", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_local_confidence_slope_hist.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def audit_files(experiment: Path) -> dict[str, list[tuple[int, Path]]]:
    found: dict[str, list[tuple[int, Path]]] = {"global": [], "local": []}
    pattern = re.compile(r"audit_(global|local)_from_R1_to_R(\d+)\.csv$")
    for path in experiment.glob("audit_*.csv"):
        match = pattern.match(path.name)
        if match:
            found[match.group(1)].append((int(match.group(2)), path))
    for model_type in found:
        found[model_type].sort(key=lambda item: item[0])
    return found


def load_audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["client_idx"] == CLIENT_ID].copy()


def save_table(df: pd.DataFrame, title: str, stem: str) -> None:
    shown = df[["alg", "auc", "low_01", "low_05", "low_1"]].copy()
    shown = shown.set_index("alg").reindex(METHODS).reset_index()
    shown.columns = ["Method", "AUC", "TPR @ 0.1% FPR", "TPR @ 0.5% FPR", "TPR @ 1% FPR"]
    display = shown.copy()
    display["AUC"] = display["AUC"].map(lambda value: f"{value:.4f}")
    for column in display.columns[2:]:
        display[column] = display[column].map(lambda value: f"{100 * value:.4f}%")

    # Export the same human-readable values that are used in the figure.  This
    # prevents spreadsheet programs from displaying long binary float tails.
    display.to_csv(FIGURES / f"{stem}.csv", index=False)

    fig_height = max(4.3, 0.43 * (len(display) + 3))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title(title, pad=18)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_tables(experiment: Path, files: dict[str, list[tuple[int, Path]]]) -> None:
    for model_type in ("local", "global"):
        round_num, csv_path = files[model_type][-1]
        df = load_audit(csv_path)
        table_name = "Table 2-style" if model_type == "local" else "Table 3-style"
        title = (
            f"{table_name}: CIFAR-10 {model_type} model auditing "
            f"(client 0, seed 0, R{round_num})"
        )
        save_table(df, title, f"{table_name.lower().replace(' ', '_')}_{model_type}_cifar10")


def series_from_files(files: list[tuple[int, Path]], metric: str) -> pd.DataFrame:
    rows = []
    for round_num, path in files:
        df = load_audit(path)
        for _, row in df.iterrows():
            rows.append({"round": round_num, "alg": row["alg"], metric: row[metric]})
    return pd.DataFrame(rows)


def make_line_chart(files: dict[str, list[tuple[int, Path]]], metric: str, title: str, stem: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, model_type in zip(axes, ("global", "local")):
        data = series_from_files(files[model_type], metric)
        for alg in METHODS:
            part = data[data["alg"] == alg].sort_values("round")
            if not part.empty:
                ax.plot(part["round"], part[metric], marker="o", linewidth=1.7, markersize=3.5, label=alg)
        ax.set(title=model_type.capitalize(), xlabel="Communication round", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(data["round"].unique()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    experiment = find_experiment()
    FIGURES.mkdir(exist_ok=True)
    files = audit_files(experiment)
    if not files["global"] or not files["local"]:
        raise FileNotFoundError("Both global and local audit CSV files are required.")

    split = find_split()
    make_figure1(experiment, split)
    make_tables(experiment, files)
    make_line_chart(
        files,
        "auc",
        "AUC across communication rounds (CIFAR-10, client 0, seed 0)",
        "auc_vs_rounds_global_local",
        "AUC",
    )
    make_line_chart(
        files,
        "low_05",
        "TPR at FPR <= 0.5% across communication rounds (CIFAR-10, client 0, seed 0)",
        "tpr_at_fpr_0_5_vs_rounds_global_local",
        "TPR at FPR <= 0.5%",
    )

    print("Finished. Created these files:")
    for path in sorted(FIGURES.iterdir()):
        print(" -", path)


if __name__ == "__main__":
    main()
