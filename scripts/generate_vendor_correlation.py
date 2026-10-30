"""Per-task accuracy of each vendor's previous model against its new model, with Spearman's rho.

One panel per vendor, points coloured by stage or by subtask grouping. Set VENDOR_PAIRS at the bottom.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from neuropsych_bench.analysis import (NINE_COLORS, STAGE_LABELS, STAGES, SUBTASK_GROUPING, SUBTASK_GROUPING_LABELS,
                                       THREE_COLORS, extract_model_name, load_test_details)


def format_p(p: float) -> str:
    """APA-style p-value."""
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".", 1)


def load_pair(new_csv: str, old_csv: str) -> pd.DataFrame:
    """Old and new accuracy per task, with its stage and subtask grouping."""
    old = pd.read_csv(old_csv)[["task", "percent_score"]].rename(columns={"percent_score": "old"})
    new = pd.read_csv(new_csv)[["task", "percent_score"]].rename(columns={"percent_score": "new"})
    unmatched = set(old["task"]) ^ set(new["task"])
    if unmatched:
        print(f"Warning: tasks not in both files were dropped: {sorted(unmatched)}")
    merged = old.merge(new, on="task").merge(load_test_details()[["task", "stage", "subtask_grouping"]],
                                             on="task", how="left")
    merged["subtask"] = merged["subtask_grouping"].map(dict(zip(SUBTASK_GROUPING, SUBTASK_GROUPING_LABELS)))
    return merged


def plot_vendor_correlation(vendor_pairs: list, output_path: str, color_by: str = "stage") -> pd.DataFrame:
    """Scatter of previous vs new accuracy per vendor; color_by is 'stage' or 'subtask'."""
    if color_by == "stage":
        categories, labels = STAGES, [STAGE_LABELS[s] for s in STAGES]
        colors = dict(zip(STAGES, THREE_COLORS))
        edge, edge_width = "white", 1
    else:
        categories, labels = SUBTASK_GROUPING_LABELS, SUBTASK_GROUPING_LABELS
        colors = dict(zip(SUBTASK_GROUPING_LABELS, NINE_COLORS))
        edge, edge_width = "0.35", 0.6  # dark outline keeps the pale yellow visible

    fig, axes = plt.subplots(1, len(vendor_pairs), figsize=(4.2 * len(vendor_pairs), 4.4), sharex=True, sharey=True)
    stats, present = [], set()

    for ax, (vendor, new_csv, old_csv) in zip(axes, vendor_pairs):
        df = load_pair(new_csv, old_csv)
        present.update(df[color_by].dropna())
        rho, p = spearmanr(df["old"], df["new"])
        stats.append({"vendor": vendor, "old_model": extract_model_name(old_csv),
                      "new_model": extract_model_name(new_csv), "n_tasks": len(df),
                      "spearman_rho": rho, "spearman_p": p, "mean_old": df["old"].mean(), "mean_new": df["new"].mean()})

        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.6", zorder=1)
        for category in categories:
            sub = df[df[color_by] == category]
            ax.scatter(sub["old"], sub["new"], s=45, color=colors[category],
                       edgecolor=edge, linewidth=edge_width, alpha=0.9, zorder=2)
        ax.set_title(f"{vendor}: {extract_model_name(old_csv)} → {extract_model_name(new_csv)}\n"
                     f"Spearman ρ = {rho:.2f} ({format_p(p)}),  n = {len(df)}", fontsize=10)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_xlabel("Previous model accuracy")
        ax.grid(color="0.92", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("New model accuracy")
    handles = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=colors[c],
                      markeredgecolor=edge, markeredgewidth=edge_width, label=label)
               for c, label in zip(categories, labels) if c in present]
    handles.append(Line2D([], [], ls="--", color="0.6", label="y = x"))
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=len(handles) if color_by == "stage" else 3,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats)


if __name__ == "__main__":
    VENDOR_PAIRS = [
        ("OpenAI", "results/results_gpt-6-sol.csv", "examples/results_gpt-4o.csv"),
        ("Anthropic", "results/results_claude-sonnet-5.csv", "examples/results_claude-3.5.csv"),
        ("Google", "results/results_gemini-3.1-pro-preview.csv", "examples/results_gemini-1.5-pro.csv"),
    ]
    stats = plot_vendor_correlation(VENDOR_PAIRS, "output/vendor_old_vs_new_correlation.png")
    plot_vendor_correlation(VENDOR_PAIRS, "output/vendor_old_vs_new_correlation_subtask.png", color_by="subtask")
    stats.to_csv("output/vendor_old_vs_new_correlation.csv", index=False)
    print(stats.round(3).to_string(index=False))
