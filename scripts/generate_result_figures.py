"""Paper figures (Supplementary Figures S2 and S3) and stage permutation tests from results CSVs.

Validated against the paper's data (https://osf.io/ysxvg/files/ck5sr). Set RESULT_CSVS at the bottom.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from neuropsych_bench.analysis import (NINE_COLORS, STAGES, SUBTASK_GROUPING_LABELS, THREE_COLORS,
                                       adjust_brightness, load_result_data, load_test_details, merge_data,
                                       prepare_data)

np.random.seed(42)
ALPHA = 0.05
N_COMPARISONS = 3  # Bonferroni correction over the three stage pairs


def permutation_test(data: pd.DataFrame, model_columns: list, n_permutations: int = 10000) -> dict:
    """Mean difference in performance relative to human between each pair of stages, with permutation p-values."""
    data["VLMs_relative_to_human"] = data[model_columns].mean(axis=1)
    stages = [data.loc[data["stage"] == s, "VLMs_relative_to_human"] for s in STAGES]
    pairs = list(combinations(range(3), 2))  # low-mid, low-high, mid-high
    observed = [stages[i].mean() - stages[j].mean() for i, j in pairs]

    sizes = np.cumsum([0] + [len(s) for s in stages])
    permuted = []
    for _ in range(n_permutations):
        shuffled = pd.concat(stages).sample(frac=1).reset_index(drop=True)
        groups = [shuffled[sizes[k]:sizes[k + 1]] for k in range(3)]
        permuted.append([groups[i].mean() - groups[j].mean() for i, j in pairs])
    permuted = np.array(permuted)

    z = (np.array(observed) - permuted.mean(axis=0)) / permuted.std(axis=0)
    p_values = scipy.stats.norm.sf(np.abs(z)) * 2
    for (i, j), diff, p in zip(pairs, observed, p_values):
        print(f"{STAGES[i]} vs {STAGES[j]}: mean difference {diff:.2f}, p = {p:.4f}, "
              f"significant after Bonferroni correction: {p < ALPHA / N_COMPARISONS}")
    return {"mean_diffs": tuple(observed), "p_values": tuple(p_values)}


def significance_marker(p: float):
    """Stars for Bonferroni-corrected significance, or None."""
    for threshold, marker in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p < threshold / N_COMPARISONS:
            return marker
    return None


def plot_stage_comparison(data: pd.DataFrame, model_columns: list, output_path: str, p_values: tuple = None):
    """Boxplot of mean performance relative to human per stage, with significance brackets."""
    data["VLMs_relative_to_human"] = data[model_columns].mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x="stage", y="VLMs_relative_to_human", data=data, hue="stage",
                palette=sns.color_palette(THREE_COLORS), width=0.6, ax=ax, legend=False)
    sns.despine()
    ax.set_xticks([0, 1, 2], ["Low", "Mid", "High"])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Averaged VLMs Performance Relative to Human")
    ax.set_ylim(-1, 0.5)

    if p_values:
        h = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        # brackets for low vs high and mid vs high
        for p, (x0, x1), y in [(p_values[1], (0, 2), h), (p_values[2], (1, 2), h - 0.08)]:
            marker = significance_marker(p)
            if marker:
                ax.plot([x0, x0, x1, x1], [y, y + 0.02, y + 0.02, y], lw=1.5, c="black")
                ax.text((x0 + x1) / 2, y + 0.03, marker, ha="center", va="bottom", color="black", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_subtask_comparison(data: pd.DataFrame, model_columns: list, output_path: str):
    """Horizontal bars of performance relative to human per test, one shade per model."""
    fig, ax = plt.subplots(figsize=(5, 15))
    color_dict = dict(zip(SUBTASK_GROUPING_LABELS, sns.color_palette(NINE_COLORS)))
    model_names = [col.removeprefix("new_").removesuffix("_human") for col in model_columns]
    brightness = [1.0 - i * 0.3 for i in range(len(model_columns))]
    bar_width = 0.3
    y_ticks, y_labels, legend_entries = [], [], {}
    bar_count = 0

    for subtask, group in data.groupby("subtask_grouping", observed=True):
        y_positions = np.arange(bar_count, bar_count + len(group))
        bar_count += len(group)
        y_ticks.extend(y_positions + bar_width)
        y_labels.extend(group["display_name"])

        for i, (model, name, factor) in enumerate(zip(model_columns, model_names, brightness)):
            bars = ax.barh(y_positions + i * bar_width, group[model], bar_width,
                           color=adjust_brightness(color_dict[subtask], factor))
            legend_entries.setdefault(f"{subtask} - {name}", bars[0])

            # * outside the bar: more than 2 SD from the human norm (bold below, regular above)
            prefix = model.removesuffix("_human")
            for bar, (_, row) in zip(bars, group.iterrows()):
                y = bar.get_y() + bar.get_height() + 0.1
                if row[f"{prefix}_sig_below"]:
                    ax.text(bar.get_width() - 0.05, y, "*", ha="center", va="bottom", color="black", fontweight="bold")
                if row[f"{prefix}_sig_above"]:
                    ax.text(bar.get_width() + 0.05, y, "*", ha="center", va="bottom", color="black")

    ax.set_yticks(y_ticks, y_labels)
    ax.set_ylabel("Test")
    ax.set_xlabel("Performance Relative to Human")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1, bar_count)
    ax.invert_yaxis()
    ax.legend(legend_entries.values(), legend_entries.keys(), bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(result_csvs: list, output_dir: str = "output", output_format: str = "svg", filename_prefix: str = ""):
    """Run the permutation tests and save the stage and subtask figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    data = prepare_data(merge_data(load_test_details(), load_result_data(result_csvs)))
    model_columns = [col for col in data.columns if col.startswith("new_") and col.endswith("_human")]
    results = permutation_test(data, model_columns)
    plot_stage_comparison(data, model_columns, output_dir / f"{prefix}stage_comparison_new.{output_format}",
                          results["p_values"])
    plot_subtask_comparison(data, model_columns, output_dir / f"{prefix}subtask_comparison_new.{output_format}")
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    RESULT_CSVS = [
        "examples/results_gpt-4o.csv",
        "examples/results_claude-3.5.csv",
        "examples/results_gemini-1.5-pro.csv",
    ]
    main(RESULT_CSVS, output_dir="output", output_format="svg")
