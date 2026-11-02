"""Longitudinal study of OpenAI models, following docs/longitudinal_inclusion_criteria.md.

    python scripts/run/longitudinal_openai.py probe   # discover models, test vision and whether reasoning can be turned off
    python scripts/run/longitudinal_openai.py run     # run all tests on every usable model (resumable)
    python scripts/run/longitudinal_openai.py plot    # performance against release date
    python scripts/run/longitudinal_openai.py trials  # rebuild trials_<model>.csv from the response logs (no API calls)

Outputs go to results/longitudinal_openai/. API key: OPENAI_KEY in .env.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # per-trial progress bars are unreadable with parallel models

import openai
import pandas as pd
from dotenv import load_dotenv

from neuropsych_bench import Evaluator, ModelConfig, OpenAIModelRunner, TaskLoader, task_names
from neuropsych_bench.analysis.longitudinal import DATED, plot_longitudinal, probe_models, run_models, summarise

OUTPUT_DIR = Path("results/longitudinal_openai")
MODELS_CSV = OUTPUT_DIR / "models.csv"
PAPER_RESULT = "examples/results_gpt-4o.csv"  # paper run of gpt-4o-2024-05-13

# OpenAI chat models only
INCLUDE = re.compile(r"^(gpt-\d|o\d)")
EXCLUDE = re.compile(r"audio|realtime|search|transcribe|tts|image|codex|instruct|live|oss|chat-latest|-pro\b|-pro-")
EFFORT_LEVELS = ["none", "minimal", "low"]  # tried lowest first

MAX_TOKENS = 1000
MAX_TOKENS_REASONING = 4000  # reasoning tokens count toward the limit


def model_tier(model: str) -> str:
    """Size class of a model; models of the same tier are connected in the plot."""
    if re.match(r"^o\d", model):
        return "o-series" + ("-mini" if "mini" in model else "")
    return next((size for size in ("mini", "nano", "luna", "terra", "astra") if f"-{size}" in model), "flagship")


def discover_models(client: openai.OpenAI) -> pd.DataFrame:
    """Chat models on the key, dropping undated aliases that have a dated snapshot."""
    models = {m.id: m.created for m in client.models.list() if INCLUDE.match(m.id) and not EXCLUDE.search(m.id)}
    dated_bases = {DATED.sub("", m) for m in models if DATED.search(m)}
    rows = [{"model": m,
             "release_date": DATED.search(m).group(1) if DATED.search(m)
             else datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d"),
             "tier": model_tier(m)}
            for m, created in models.items() if DATED.search(m) or m not in dated_bases]
    return pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)


def make_runner(model: str, max_tokens: int, params: dict) -> OpenAIModelRunner:
    """Runner at the provider's default temperature, with extra retries for long runs."""
    runner = OpenAIModelRunner(ModelConfig(model_name=model, api_key=os.getenv("OPENAI_KEY"), max_tokens=max_tokens,
                                           temperature=None, additional_params=params))
    runner.client = runner.client.with_options(max_retries=8)
    return runner


def describe(response) -> dict:
    """Token usage and truncation of one response."""
    details = response.usage.completion_tokens_details
    return {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens,
            "reasoning_tokens": (details.reasoning_tokens if details else 0) or 0,
            "truncated": response.choices[0].finish_reason == "length"}


def probe_model(model: str, loader: TaskLoader) -> dict:
    """Check the model answers a real trial, then find the lowest reasoning effort it accepts."""
    trial = loader.get_trials()[0]

    def reasoning_tokens(params):
        return describe(make_runner(model, MAX_TOKENS_REASONING, params).run_trial(trial))["reasoning_tokens"]

    try:
        default_reasoning = reasoning_tokens({})
    except openai.OpenAIError as e:
        return {"usable": False, "reasoning_mode": "unusable", "reasoning_effort": None,
                "flagged": None, "note": f"{type(e).__name__}: {str(e)[:150]}"}

    for effort in EFFORT_LEVELS:
        try:
            reasoning = reasoning_tokens({"reasoning_effort": effort})
        except openai.BadRequestError:
            continue
        off = effort == "none" and reasoning == 0
        return {"usable": True, "reasoning_mode": "off" if off else f"lowest={effort}",
                "reasoning_effort": effort, "flagged": not off,
                "note": f"default reasoning tokens={default_reasoning}, at {effort}={reasoning}"}

    if default_reasoning == 0:
        return {"usable": True, "reasoning_mode": "non-reasoning model", "reasoning_effort": None,
                "flagged": False, "note": "no reasoning_effort parameter"}
    return {"usable": True, "reasoning_mode": "cannot be lowered", "reasoning_effort": None, "flagged": True,
            "note": f"default reasoning tokens={default_reasoning}; reasoning_effort rejected"}


def runner_for(row: pd.Series) -> OpenAIModelRunner:
    """Runner with the reasoning setting and token limit chosen by the probe."""
    params = {"reasoning_effort": row["reasoning_effort"]} if isinstance(row["reasoning_effort"], str) else {}
    return make_runner(row["model"], MAX_TOKENS_REASONING if row["flagged"] else MAX_TOKENS, params)


def rebuild_trials():
    """Re-score every response log into trials_<model>.csv and check the scores match results_<model>.csv."""
    loaders = {task: TaskLoader(task) for task in task_names()}
    for log_path in sorted(OUTPUT_DIR.glob("responses_*.jsonl")):
        model = log_path.stem.removeprefix("responses_")
        if not (OUTPUT_DIR / f"results_{model}.csv").exists():  # run stopped before its first task finished
            print(f"{model}: no results yet, skipped")
            continue
        with open(log_path, encoding="utf-8") as log:
            responses = {(r["task"], r["trial"]): r["response"] for r in map(json.loads, log)}

        evaluator = Evaluator()
        for task, loader in loaders.items():
            trials = loader.get_trials()
            if not all((task, trial["trial_id"]) in responses for trial in trials):
                continue
            for trial in trials:
                trial["model_response"] = responses[(task, trial["trial_id"])]
            evaluator.evaluate((loader.get_task_info(), trials))
        evaluator.save_trials_as_csv(OUTPUT_DIR / f"trials_{model}.csv")

        saved = pd.read_csv(OUTPUT_DIR / f"results_{model}.csv").set_index("task")["raw_score"]
        rescored = evaluator.get_result().set_index("task")["raw_score"]
        mismatched = [t for t in saved.index if rescored.get(t) != saved[t]]
        print(f"{model}: {len(evaluator.get_trials())} trials, {len(rescored)}/{len(saved)} tasks, "
              f"{'OK' if not mismatched else f'score mismatch on {mismatched}'}")


def plot(output_format: str = "svg"):
    """Plot performance against release date and save the summary table."""
    table = summarise(MODELS_CSV, OUTPUT_DIR, [PAPER_RESULT])
    paper = ("gpt-4o-2024-05-13", table.attrs["data"]["new_gpt-4o_human"].mean())
    plot_longitudinal(table, "OpenAI", Path("output") / f"longitudinal_openai.{output_format}", paper)
    table.round(3).to_csv("output/longitudinal_openai_summary.csv", index=False)
    print(table[["model", "release_date", "tier", "reasoning_mode", "accuracy", "rel_human"]].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["probe", "run", "plot", "trials"])
    parser.add_argument("--workers", type=int, default=4, help="models processed in parallel")
    parser.add_argument("--models", nargs="*", help="run: only these models (default: all usable)")
    parser.add_argument("--skip-flagged", action="store_true", help="run: only models with reasoning off")
    parser.add_argument("--format", default="svg", choices=["svg", "png", "pdf"], help="plot: output format")
    args = parser.parse_args()
    load_dotenv()

    if args.step == "probe":
        models = discover_models(openai.OpenAI(api_key=os.getenv("OPENAI_KEY")))
        probe_models(models, partial(probe_model, loader=TaskLoader(task_names()[0])), MODELS_CSV, args.workers)
    elif args.step == "run":
        run_models(MODELS_CSV, runner_for, describe, OUTPUT_DIR, args.workers, args.models, args.skip_flagged)
    elif args.step == "trials":
        rebuild_trials()
    else:
        plot(args.format)
