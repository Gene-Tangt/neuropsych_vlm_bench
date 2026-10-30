"""Performance relative to human per test for new models, each overlaid with its vendor's previous model.

Same metric and style as generate_result_figures.py. Set VENDOR_PAIRS at the bottom.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from neuropsych_bench.analysis import (NINE_COLORS, STAGE_LABELS, STAGES, SUBTASK_GROUPING_LABELS, THREE_COLORS,
                                       adjust_brightness, extract_model_name, load_result_data, load_test_details,
                                       merge_data, prepare_data)

SUBTASK_PANEL_LABELS = [
    "Simple\nelement\njudgement",
    "Figure\njudgement",
    "Occlusion\nand overlap",
    "Property\nbiases",
    "Robustness\nacross\nimage cues",
    "Robustness\nacross different\nconfigurations",
    "Straightforward\nvisual recognition",
    "View\ninvariance",
    "Semantic association\nand categorisation",
]
MODEL_GREYS = ["#d9d9d9", "#8c8c8c", "#404040"]  # legend swatches for each model's shade


def human_col(csv_path: str) -> str:
    """Column of prepare_data holding performance relative to human for a results CSV."""
    return f"new_{extract_model_name(csv_path)}_human"


def plot_vendor_comparison(data: pd.DataFrame, vendor_pairs: list, output_path: str):
    """Bars for new models with an outline for the previous model, on subtask bands with a stage strip."""
    fig, ax = plt.subplots(figsize=(6, 15))
    color_dict = dict(zip(SUBTASK_GROUPING_LABELS, sns.color_palette(NINE_COLORS)))
    panel_labels = dict(zip(SUBTASK_GROUPING_LABELS, SUBTASK_PANEL_LABELS))
    stage_colors = dict(zip(STAGES, sns.color_palette(THREE_COLORS)))

    num_models = len(vendor_pairs)
    bar_width = 0.3
    band_gap = 0.04
    y_ticks, y_labels, stage_extents = [], [], {}
    bar_count = 0

    for subtask, group in data.groupby("subtask_grouping", observed=True):
        y_positions = np.arange(bar_count, bar_count + len(group))
        bar_count += len(group)
        y_ticks.extend(y_positions + bar_width)
        y_labels.extend(group["display_name"])

        band_top = y_positions[0] - bar_width / 2 - 0.05 + band_gap
        band_bottom = y_positions[-1] + (num_models - 0.5) * bar_width + 0.05 - band_gap
        ax.axhspan(band_top, band_bottom, color=color_dict[subtask], alpha=0.25, linewidth=0, zorder=0)
        ax.text(1.07, band_top + 0.12, panel_labels[subtask], ha="right", va="top",
                color=adjust_brightness(color_dict[subtask], 0.8), fontsize=9, fontweight="bold", zorder=1)

        stage = group["stage"].iloc[0]
        top, bottom = stage_extents.get(stage, (band_top, band_bottom))
        stage_extents[stage] = (min(top, band_top), max(bottom, band_bottom))

        for i, (_, new_csv, old_csv) in enumerate(vendor_pairs):
            y = y_positions + i * bar_width
            bars = ax.barh(y, group[human_col(new_csv)], bar_width,
                           color=adjust_brightness(color_dict[subtask], 1.0 - i * 0.3), zorder=2)
            ax.barh(y, group[human_col(old_csv)], bar_width, fill=False, edgecolor="black", linewidth=0.8, zorder=3)

            # * outside the bar: new model more than 2 SD from the human norm
            prefix = human_col(new_csv).removesuffix("_human")
            for bar, (_, row) in zip(bars, group.iterrows()):
                y_text = bar.get_y() + bar.get_height() + 0.1
                if row[f"{prefix}_sig_below"]:
                    ax.text(bar.get_width() - 0.05, y_text, "*", ha="center", va="bottom", color="black",
                            fontweight="bold", zorder=4)
                if row[f"{prefix}_sig_above"]:
                    ax.text(bar.get_width() + 0.05, y_text, "*", ha="center", va="bottom", color="black", zorder=4)

    ax.axvline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_yticks(y_ticks, y_labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Performance Relative to Human")
    ax.set_xlim(-1.1, 1.1)
    ax.set_xticks(np.arange(-1, 1.01, 0.25))
    ax.set_ylim(min(t for t, _ in stage_extents.values()), max(b for _, b in stage_extents.values()))
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)

    strip = ax.inset_axes([1.02, 0, 0.06, 1], sharey=ax)
    for stage, (top, bottom) in stage_extents.items():
        strip.axhspan(top, bottom, color=stage_colors[stage], linewidth=0)
        strip.text(0.5, (top + bottom) / 2, STAGE_LABELS[stage], rotation=270,
                   ha="center", va="center", color="white", fontsize=10)
    strip.set_xlim(0, 1)
    strip.axis("off")

    handles = [Patch(facecolor=grey, edgecolor="none") for grey in MODEL_GREYS[:num_models]]
    labels = [extract_model_name(new) for _, new, _ in vendor_pairs]
    handles.append(Patch(fill=False, edgecolor="black", linewidth=0.8))
    labels.append("Previous, same vendor\n(" + ", ".join(extract_model_name(old) for _, _, old in vendor_pairs) + ")")
    ax.legend(handles, labels, title="Models", loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              ncol=2, fontsize=8, title_fontsize=9, frameon=False, alignment="left")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def summarise(data: pd.DataFrame, vendor_pairs: list, output_path: str):
    """Save per-test previous vs new values and print the mean change per vendor."""
    table = data[["task", "display_name", "stage", "subtask_grouping"]].copy()
    for vendor, new_csv, old_csv in vendor_pairs:
        new_name, old_name = extract_model_name(new_csv), extract_model_name(old_csv)
        new_vals, old_vals = data[human_col(new_csv)], data[human_col(old_csv)]
        table[f"{vendor}_previous ({old_name})"] = old_vals.round(3)
        table[f"{vendor}_new ({new_name})"] = new_vals.round(3)
        table[f"{vendor}_change"] = (new_vals - old_vals).round(3)
        print(f"{vendor}: {old_name} {old_vals.mean():+.2f} -> {new_name} {new_vals.mean():+.2f} "
              f"(mean relative to human), {(new_vals > old_vals).sum()}/{len(data)} tests improved")
    table.to_csv(output_path, index=False)


def main(vendor_pairs: list, output_dir: str = "output", output_format: str = "svg"):
    """Save the comparison figure and table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csvs = [path for _, new, old in vendor_pairs for path in (new, old)]
    data = prepare_data(merge_data(load_test_details(), load_result_data(csvs)))
    plot_vendor_comparison(data, vendor_pairs, output_dir / f"subtask_comparison_vs_previous.{output_format}")
    summarise(data, vendor_pairs, output_dir / "vendor_comparison_table.csv")


if __name__ == "__main__":
    # (vendor, new model results, previous results of the same vendor)
    VENDOR_PAIRS = [
        ("OpenAI", "results/results_gpt-6-sol.csv", "examples/results_gpt-4o.csv"),
        ("Anthropic", "results/results_claude-sonnet-5.csv", "examples/results_claude-3.5.csv"),
        ("Google", "results/results_gemini-3.1-pro-preview.csv", "examples/results_gemini-1.5-pro.csv"),
    ]
    main(VENDOR_PAIRS, output_dir="output", output_format="svg")
