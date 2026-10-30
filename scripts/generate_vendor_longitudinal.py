"""Performance relative to human over release date for the OpenAI flagship and Claude Sonnet lines.

Same metric and style as longitudinal_openai.py plot. Reads results/longitudinal_*/ and writes to output/.

    python scripts/generate_vendor_longitudinal.py [--format svg|png|pdf]
"""

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from neuropsych_bench.analysis import STAGES, THREE_COLORS, extract_model_name
from neuropsych_bench.longitudinal import MARKER_SIZE, short_name, summarise

def human_col(csv_path: str) -> str:
    """Column of prepare_data holding performance relative to human for a results CSV."""
    return f"new_{extract_model_name(csv_path)}_human"


MAIN_LINES = json.loads((Path(__file__).parent / "main_line_models.json").read_text())
PAPER = MAIN_LINES["paper_runs"]
PAPER_RESULT = PAPER["models"]["gpt-4o-2024-05-13"]["results"]
# claude-3-5-sonnet-20241022 is retired, so only its paper run exists
_claude = PAPER["models"]["claude-3-5-sonnet-20241022"]
CLAUDE_PAPER = (_claude["results"], human_col(_claude["results"]), pd.Timestamp(_claude["release_date"]))


def line(key: str, color: str, marker: str, style: str) -> tuple:
    """(label, results folder, models, colour, marker, line style) of a line in main_line_models.json."""
    entry = MAIN_LINES[key]
    return entry["label"], Path(entry["results_dir"]), list(entry["models"]), color, marker, style


LINES = [line("openai_flagship", "#10a37f", "o", "-"), line("claude_sonnet", "#d97757", "s", "--")]


def load_line(output_dir: Path, models: list, extra_results: list = ()) -> pd.DataFrame:
    """Summary of the listed models, sorted by release date; every listed model must be fully run."""
    table = summarise(output_dir / "models.csv", output_dir, extra_results)
    data = table.attrs["data"]
    missing = set(models) - set(table["model"])
    if missing:
        raise ValueError(f"Not fully run in {output_dir}: {sorted(missing)}")
    table = (table[table["model"].isin(models)].assign(date=lambda t: pd.to_datetime(t["release_date"]))
             .sort_values("date"))
    table.attrs["data"] = data
    return table


def scatter(ax, date, value, color, marker):
    return ax.scatter(date, value, s=MARKER_SIZE, marker=marker, facecolors=color, edgecolors="white", linewidths=1,
                      zorder=3)


def plot_paper_point(ax, date, value, color, marker):
    """Hollow, unconnected marker for a paper run that was not rerun."""
    ax.scatter(date, value, s=MARKER_SIZE, marker=marker, facecolors="white", edgecolors=color, linewidths=1.5,
               zorder=3)


def plot(output_path: Path):
    paper_results = {"OpenAI flagship": [PAPER_RESULT], "Claude Sonnet": [CLAUDE_PAPER[0]]}
    tables = {label: load_line(output_dir, models, paper_results[label]) for label, output_dir, models, *_ in LINES}
    claude = tables["Claude Sonnet"]
    claude_data = claude.attrs["data"]
    claude_paper = {"rel_human": claude_data[CLAUDE_PAPER[1]].mean(),
                    **{f"rel_human_{s}": claude_data.loc[claude_data["stage"] == s, CLAUDE_PAPER[1]].mean()
                       for s in STAGES}}
    claude_color, claude_marker = LINES[1][3], LINES[1][4]
    if any(table["flagged"].astype(bool).any() for table in tables.values()):
        raise ValueError("Some models could only have reasoning minimised; this figure assumes reasoning off")
    fig = plt.figure(figsize=(10, 9))
    grid = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.08)
    ax_all = fig.add_subplot(grid[0, :])
    ax_stages = [fig.add_subplot(grid[1, 0])]
    ax_stages += [fig.add_subplot(grid[1, i], sharex=ax_stages[0], sharey=ax_stages[0]) for i in (1, 2)]

    for i, (label, _, _, color, marker, style) in enumerate(LINES):
        table = tables[label]
        all_models = list(table["model"])
        # labels above for the first line, below for the second, so close releases don't collide
        above = i == 0
        ax_all.plot(table["date"], table["rel_human"], color=color, linestyle=style, linewidth=1.5, label=label,
                    zorder=2)
        for _, row in table.iterrows():
            scatter(ax_all, row["date"], row["rel_human"], color, marker)
            ax_all.annotate(short_name(row["model"], all_models), (row["date"], row["rel_human"]), xytext=(5, 6 if above else -7),
                            textcoords="offset points", va="bottom" if above else "top", fontsize=6.5,
                            rotation=25 if above else -25, rotation_mode="anchor", color="0.3")

    # outline only, so the rerun of the same model stays visible underneath
    openai = tables["OpenAI flagship"]
    paper_value = openai.attrs["data"][human_col(PAPER_RESULT)].mean()
    ax_all.scatter(openai.loc[openai["model"] == "gpt-4o-2024-05-13", "date"].iloc[0], paper_value, marker="*",
                   s=200, facecolors="none", edgecolors="black", linewidths=1, zorder=4)
    plot_paper_point(ax_all, CLAUDE_PAPER[2], claude_paper["rel_human"], claude_color, claude_marker)
    ax_all.annotate("claude-3.5-sonnet", (CLAUDE_PAPER[2], claude_paper["rel_human"]), xytext=(5, -7),
                    textcoords="offset points", va="top", fontsize=6.5, rotation=-25, rotation_mode="anchor",
                    color="0.3")
    handles = [Line2D([], [], color=color, linestyle=style, marker=marker, markeredgecolor="white")
               for _, _, _, color, marker, style in LINES]
    handles.append(Line2D([], [], marker="*", linestyle="", markerfacecolor="none", markeredgecolor="black",
                          markersize=12))
    handles.append(Line2D([], [], linestyle="", marker=claude_marker, markerfacecolor="white",
                          markeredgecolor=claude_color, markeredgewidth=1.5))
    labels = [entry[0] for entry in LINES] + [f"gpt-4o (05-13),\n{PAPER['citation']}",
                                              f"claude-3.5-sonnet (10-22),\n{PAPER['citation']}"]
    ax_all.axhline(0, color="black", linewidth=0.6)
    ax_all.set_ylabel("Mean performance relative to human")
    ax_all.set_title("(a) OpenAI flagship and Claude Sonnet models with reasoning off or absent", loc="left",
                     fontsize=10, fontweight="bold", color="black")
    ax_all.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.07, 1), fontsize=8, frameon=False)

    # one panel per stage, same vendor colours as (a)
    # one panel letter for the whole row, then each stage name in its colour
    letter = ax_stages[0].text(0, 1.02, "(b)", transform=ax_stages[0].transAxes, fontsize=10, fontweight="bold",
                               va="bottom")
    for ax, stage, stage_color in zip(ax_stages, STAGES, THREE_COLORS):
        for label, _, _, color, marker, style in LINES:
            table = tables[label]
            ax.plot(table["date"], table[f"rel_human_{stage}"], color=color, linestyle=style, linewidth=1.5, zorder=2)
            scatter(ax, table["date"], table[f"rel_human_{stage}"], color, marker)
        plot_paper_point(ax, CLAUDE_PAPER[2], claude_paper[f"rel_human_{stage}"], claude_color, claude_marker)
        ax.axhline(0, color="black", linewidth=0.6)
        title = dict(fontsize=10, fontweight="bold", color=stage_color, va="bottom")
        if ax is ax_stages[0]:
            ax.annotate(f"{stage.capitalize()}-level", xy=(1, 0), xycoords=letter, xytext=(4, 0),
                        textcoords="offset points", **title)
        else:
            ax.text(0, 1.02, f"{stage.capitalize()}-level", transform=ax.transAxes, **title)
    for ax in ax_stages[1:]:
        ax.tick_params(labelleft=False)
    ax_stages[0].set_ylabel("Mean performance relative to human")
    ax_stages[1].set_xlabel("Model release date")
    # narrow panels: label January only, with unlabelled quarterly ticks
    ax_stages[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax_stages[0].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
    ax_stages[0].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))  # e.g. Jan-25

    ax_all.set_xlim(ax_stages[0].get_xlim())
    ax_all.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax_all.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    for ax in (ax_all, *ax_stages):
        ax.spines[["top", "right"]].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    for label, table in tables.items():
        print(label)
        print(table[["model", "release_date", "rel_human"] + [f"rel_human_{s}" for s in STAGES]].round(3)
              .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", default="svg", choices=["svg", "png", "pdf"], help="output format")
    args = parser.parse_args()
    plot(Path("output") / f"longitudinal_openai_claude.{args.format}")
