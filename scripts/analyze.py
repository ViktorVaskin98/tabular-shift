import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLIT_ORDER = [
    "row-random",
    "group-random",
    "cluster-0",
    "cluster-1",
    "cluster-2",
    "no-iron",
    "no-cuprate",
    "tc-extrapolation",
]
MODEL_ORDER = ["ridge", "catboost", "tabicl", "tabdpt"]
COLORS = {"ridge": "#b0b0b0", "catboost": "#e07a5f", "tabicl": "#2a6f97", "tabdpt": "#3d405b"}
LABELS = {"ridge": "Ridge", "catboost": "CatBoost", "tabicl": "TabICL", "tabdpt": "TabDPT"}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rmse", "spearman", "precision_at_50", "discovery_at_50", "max_pred"]
    grouped = raw.groupby(["split", "model"])[metrics]
    table = grouped.mean().join(grouped.std().add_suffix("_sd")).reset_index()
    table["split"] = pd.Categorical(table["split"], SPLIT_ORDER, ordered=True)
    table["model"] = pd.Categorical(table["model"], MODEL_ORDER, ordered=True)
    return table.sort_values(["split", "model"])


STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def plot_metric(
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out: Path,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> None:
    plt.rcParams.update(STYLE)
    lower, upper = bounds
    splits = [s for s in SPLIT_ORDER if s in set(table["split"].astype(str))]
    models = [m for m in MODEL_ORDER if m in set(table["model"].astype(str))]
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(13, 5.4))
    for i, model in enumerate(models):
        rows = table[table["model"] == model].set_index("split").reindex(splits)
        x = np.arange(len(splits)) + (i - (len(models) - 1) / 2) * width
        ax.bar(
            x,
            rows[metric],
            width,
            yerr=[
                np.minimum(rows[f"{metric}_sd"], rows[metric] - lower),
                np.minimum(rows[f"{metric}_sd"], upper - rows[metric]),
            ],
            capsize=3,
            error_kw={"elinewidth": 1.2},
            color=COLORS[model],
            label=LABELS[model],
        )
    ax.set_xticks(np.arange(len(splits)), splits, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=34)
    ax.axvline(1.5, color="grey", linewidth=1, linestyle=":")
    top = ax.get_ylim()[1]
    ax.text(0.5, top, "IID", ha="center", va="bottom", fontsize=12, color="grey")
    ax.text(
        (len(splits) + 1) / 2,
        top,
        "distribution shift",
        ha="center",
        va="bottom",
        fontsize=12,
        color="grey",
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tabular-shift results")
    parser.add_argument("raw", type=Path, nargs="?", default=Path("results/raw.csv"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    table = summarize(raw)
    args.results.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results / "summary.csv", index=False)
    plot_metric(
        table,
        "spearman",
        "Spearman ρ (mean ± sd over seeds)",
        "Ranking quality across train/test splits",
        args.figures / "spearman_by_split.png",
    )
    plot_metric(
        table,
        "precision_at_50",
        "Precision@50 for top-10% Tc",
        "Screening: true top-10% materials among each model's 50 best candidates",
        args.figures / "screening_by_split.png",
    )
    extrapolation = raw[raw["split"] == "tc-extrapolation"]
    print(table.round(3).to_string(index=False))
    print("\ntc-extrapolation: max prediction vs training ceiling")
    print(
        extrapolation.groupby("model")[["max_pred", "max_train_target", "discovery_at_50"]]
        .mean()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
