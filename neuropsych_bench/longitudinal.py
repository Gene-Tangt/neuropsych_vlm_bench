import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .analysis import STAGES, THREE_COLORS, load_result_data, load_test_details, merge_data, prepare_data
from .evaluator import Evaluator
from .loaders import TaskLoader, task_paths

DATED = re.compile(r"-(\d{4}-\d{2}-\d{2})$")
SNAPSHOT = re.compile(r"-(\d{4})-?(\d{2})-?(\d{2})$")  # 2024-05-13 (OpenAI) or 20241022 (Anthropic)
SLEEP_BETWEEN_TASKS = 2  # seconds
MARKER_SIZE = 64


def short_name(model: str, all_models: list) -> str:
    """Model id without its snapshot date, unless several snapshots share the name (then keep MM-DD)."""
    base = SNAPSHOT.sub("", model)
    if base == model or sum(SNAPSHOT.sub("", m) == base for m in all_models) == 1:
        return base
    _, month, day = SNAPSHOT.search(model).groups()
    return f"{base} ({month}-{day})"


def probe_models(models: pd.DataFrame, probe: Callable, models_csv: Path, workers: int):
    """Add the probe result of every model to the table and save it."""
    print(f"Probing {len(models)} models...")

    def safe_probe(model):
        try:
            return probe(model)
        except Exception as e:  # one failing model must not stop the others; rerun the probe to retry it
            return {"usable": False, "reasoning_mode": "probe failed", "flagged": None,
                    "note": f"{type(e).__name__}: {str(e)[:150]}"}

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(safe_probe, models["model"]))
    models = pd.concat([models, pd.DataFrame(results)], axis=1)
    models_csv.parent.mkdir(parents=True, exist_ok=True)
    models.to_csv(models_csv, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(models.drop(columns="note").to_string(index=False))
    print(f"\nSaved to {models_csv}")


def run_model(runner, output_dir: Path, describe: Callable):
    """Run every task on one model, saving after each task; describe(api_response) gives the fields logged per trial."""
    model = runner.config.model_name
    paths = {kind: output_dir / f"{kind}_{model}.{ext}" for kind, ext in
             [("results", "csv"), ("trials", "csv"), ("usage", "json"), ("responses", "jsonl")]}
    all_tasks = task_paths()

    evaluator = Evaluator()
    done = set()
    if paths["results"].exists():
        evaluator.result_table = pd.read_csv(paths["results"])
        done = set(evaluator.result_table["task"])
        if paths["trials"].exists():
            evaluator.trial_table = pd.read_csv(paths["trials"])
    if len(done) == len(all_tasks):
        return
    usage = json.loads(paths["usage"].read_text()) if paths["usage"].exists() and done else {"calls": 0}

    with open(paths["responses"], "a" if done else "w", encoding="utf-8") as log:
        for task_path in all_tasks:
            loader = TaskLoader(task_path)
            if loader.get_task_info()["task"] in done:
                continue
            info, trials = runner.generate_response(loader)
            evaluator.evaluate((info, trials))
            for trial in trials:
                fields = describe(trial["api_response"])  # numeric and boolean fields are summed into usage
                usage["calls"] += 1
                for key, value in fields.items():
                    if isinstance(value, (int, float)):
                        usage[key] = usage.get(key, 0) + value
                log.write(json.dumps({"task": info["task"], "trial": trial["trial_id"],
                                      "response": trial["model_response"], **fields}) + "\n")
            log.flush()
            evaluator.save_as_csv(paths["results"])
            evaluator.save_trials_as_csv(paths["trials"])
            paths["usage"].write_text(json.dumps(usage))
            last = evaluator.get_result().iloc[-1]
            print(f"[{model}] {len(evaluator.get_result())}/{len(all_tasks)} {last['task']}: "
                  f"{last['raw_score']}/{last['num_trials']}", flush=True)
            time.sleep(SLEEP_BETWEEN_TASKS)


def run_models(models_csv: Path, make_runner: Callable, describe: Callable, output_dir: Path, workers: int,
               only: list = None, skip_flagged: bool = False):
    """Run every usable model of a probe table in parallel; make_runner(row) builds each model's runner."""
    models = pd.read_csv(models_csv)
    models = models[models["usable"]]
    if skip_flagged:
        models = models[models["flagged"] == False]  # noqa: E712 (booleans read from CSV)
    if only:
        models = models[models["model"].isin(only)]

    def safe_run(row):
        try:
            run_model(make_runner(row), output_dir, describe)
            print(f"[{row['model']}] DONE", flush=True)
        except Exception as e:  # one failing model must not stop the others; a rerun resumes it
            print(f"[{row['model']}] STOPPED: {type(e).__name__}: {str(e)[:200]}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(safe_run, [row for _, row in models.iterrows()]))


def summarise(models_csv: Path, output_dir: Path, extra_results: list = ()) -> pd.DataFrame:
    """Accuracy and performance relative to human (overall and per stage) of every fully run model."""
    models = pd.read_csv(models_csv)
    models = models[models["usable"]]
    paths = {m: output_dir / f"results_{m}.csv" for m in models["model"]}
    complete = [m for m, p in paths.items() if p.exists() and len(pd.read_csv(p)) == len(task_paths())]
    models = models[models["model"].isin(complete)]
    data = prepare_data(merge_data(load_test_details(),
                                   load_result_data([str(paths[m]) for m in complete] + list(extra_results))))

    rows = []
    for _, row in models.iterrows():
        m = row["model"]
        usage_path = output_dir / f"usage_{m}.json"
        usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
        rows.append({**row.to_dict(), "accuracy": data[f"new_{m}_score_percent"].mean(),
                     "rel_human": data[f"new_{m}_human"].mean(), "truncated_responses": usage.get("truncated"),
                     **{f"rel_human_{s}": data.loc[data["stage"] == s, f"new_{m}_human"].mean() for s in STAGES}})
    table = pd.DataFrame(rows)
    table.attrs["data"] = data
    return table


def scatter_model(ax, date, value, color, flagged: bool):
    """Solid dot when reasoning was off (or absent); dotted outline when it could only be minimised."""
    if flagged:
        return ax.scatter(date, value, s=MARKER_SIZE, facecolors="white", edgecolors=color, linewidths=1.5,
                          linestyles=(0, (1.2, 1.2)), zorder=3)
    return ax.scatter(date, value, s=MARKER_SIZE, facecolors=color, edgecolors="white", linewidths=1, zorder=3)


def plot_longitudinal(table: pd.DataFrame, provider: str, output_path: Path, paper_point: tuple = None):
    """Performance relative to human over release date by tier and, for flagships, by stage; paper_point is (model, value)."""
    table = table.assign(date=pd.to_datetime(table["release_date"])).sort_values("date")
    fig, (ax_all, ax_stage) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    tier_colors = dict(zip(table["tier"].unique(), plt.get_cmap("tab10").colors))
    all_models = list(table["model"])

    for tier, group in table.groupby("tier", sort=False):
        color = tier_colors[tier]
        ax_all.plot(group["date"], group["rel_human"], color=color, linewidth=1.5, label=tier, zorder=2)
        # labels above for flagship and terra, below for the rest, so same-day releases don't collide
        above = tier in ("flagship", "terra")
        for _, row in group.iterrows():
            scatter_model(ax_all, row["date"], row["rel_human"], color, row["flagged"])
            ax_all.annotate(short_name(row["model"], all_models), (row["date"], row["rel_human"]),
                            xytext=(5, 6 if above else -7), textcoords="offset points",
                            va="bottom" if above else "top", fontsize=6.5, rotation=25 if above else -25,
                            rotation_mode="anchor", color="0.3")

    # marker legend only when both kinds are plotted; otherwise the titles say which kind it is
    kinds = set(table["flagged"].astype(bool))
    scope = "" if len(kinds) == 2 else " with reasoning minimised" if True in kinds else " with reasoning off or absent"
    handles, labels = ax_all.get_legend_handles_labels()
    if len(kinds) == 2:
        handles += [scatter_model(ax_all, [], [], "0.3", flagged) for flagged in (False, True)]
        labels += ["reasoning turned off\n(or no reasoning)", "reasoning could only\nbe minimised"]
    if paper_point and paper_point[0] in all_models:
        # outline only, so the rerun of the same model stays visible underneath
        ax_all.scatter(table.loc[table["model"] == paper_point[0], "date"].iloc[0], paper_point[1], marker="*",
                       s=200, facecolors="none", edgecolors="black", linewidths=1, zorder=4)
        handles.append(Line2D([], [], marker="*", linestyle="", markerfacecolor="none", markeredgecolor="black",
                              markersize=12))
        labels.append(f"{short_name(paper_point[0], all_models)}, original paper run")
    ax_all.axhline(0, color="black", linewidth=0.6)
    ax_all.set_ylabel("Mean performance relative to human")
    ax_all.set_title(f"(a) {provider} models{scope}", loc="left", fontsize=10)
    ax_all.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)

    flagship = table[table["tier"] == "flagship"]
    for stage, color in zip(STAGES, THREE_COLORS):
        ax_stage.plot(flagship["date"], flagship[f"rel_human_{stage}"], color=color, linewidth=1.5,
                      label=f"{stage}-level", zorder=2)
        for _, row in flagship.iterrows():
            scatter_model(ax_stage, row["date"], row[f"rel_human_{stage}"], color, row["flagged"])
    ax_stage.axhline(0, color="black", linewidth=0.6)
    ax_stage.set_ylabel("Mean performance relative to human")
    ax_stage.set_xlabel("Model release date")
    ax_stage.set_title(f"(b) Flagship {provider} models{scope}, by processing stage", loc="left", fontsize=10)
    ax_stage.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    for ax in (ax_all, ax_stage):
        ax.spines[["top", "right"]].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
