"""Longitudinal study of flagship Gemini models, following docs/longitudinal_inclusion_criteria.md.

    python scripts/longitudinal_gemini.py probe   # find flagship models, test vision and whether thinking can be turned off
    python scripts/longitudinal_gemini.py run     # run all tests on every usable model (resumable)
    python scripts/longitudinal_gemini.py plot    # performance against release date

Outputs go to results/longitudinal_gemini/. API key: GEMINI_KEY in .env.
"""

import argparse
import json
import os
import re
from functools import partial
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # per-trial progress bars are unreadable with parallel models

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from neuropsych_bench import GoogleModelRunner, ModelConfig, TaskLoader, task_paths
from neuropsych_bench.longitudinal import plot_longitudinal, probe_models, run_models, summarise

OUTPUT_DIR = Path("results/longitudinal_gemini")
MODELS_CSV = OUTPUT_DIR / "models.csv"

FLAGSHIP = re.compile(r"^gemini-(\d+(\.\d+)?-pro(-preview)?|\d+-argon)$")
EXTRA_CANDIDATES = ["gemini-3-pro-preview"]  # not listed by the API but may still be served
# The Gemini API does not report release dates
RELEASE_DATES = {
    "gemini-2.5-pro": "2025-06-17",
    "gemini-3-pro-preview": "2025-11-18",
    "gemini-3.1-pro-preview": "2026-02-19",
    "gemini-4-argon": "2026-09-30",
}
THINKING_OFF = {"thinking_config": {"thinking_budget": 0}}
LOWEST_THINKING = [{"thinking_config": {"thinking_level": "minimal"}}, {"thinking_config": {"thinking_level": "low"}},
                   {"thinking_config": {"thinking_budget": 128}}]  # tried in this order; 128 is 2.5 Pro's minimum

MAX_TOKENS = 1000
MAX_TOKENS_REASONING = 4000  # thinking tokens count toward the limit


def discover_models(client: genai.Client) -> pd.DataFrame:
    """Flagship Gemini models on the key, plus candidates the API no longer lists."""
    listed = [m.name.removeprefix("models/") for m in client.models.list()]
    models = sorted({m for m in listed if FLAGSHIP.match(m)} | set(EXTRA_CANDIDATES))
    return pd.DataFrame([{"model": m, "release_date": RELEASE_DATES.get(m), "tier": "flagship"} for m in models])


def make_runner(model: str, max_tokens: int, params: dict) -> GoogleModelRunner:
    """Runner at the provider's default temperature, with extra retries for long runs."""
    runner = GoogleModelRunner(ModelConfig(model_name=model, api_key=os.getenv("GEMINI_KEY"), max_tokens=max_tokens,
                                           temperature=None, additional_params=params))
    runner.client = genai.Client(api_key=os.getenv("GEMINI_KEY"), http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=8)))
    return runner


def describe(response) -> dict:
    """Token usage, finish reason and whether the response contains thinking."""
    usage = response.usage_metadata
    finish = response.candidates[0].finish_reason if response.candidates else None
    thinking_tokens = usage.thoughts_token_count or 0
    return {"input_tokens": usage.prompt_token_count or 0, "output_tokens": usage.candidates_token_count or 0,
            "thinking_tokens": thinking_tokens, "thinking": thinking_tokens > 0,
            "truncated": finish == types.FinishReason.MAX_TOKENS,
            "finish_reason": finish.name if finish else "NO_CANDIDATE"}


def probe_model(model: str, loader: TaskLoader) -> dict:
    """Check the model answers a real trial, then whether thinking can be turned off or lowered."""
    trial, num_stim = loader.get_trials()[0], loader.get_task_info()["num_stim"]

    def thinking_tokens(params):
        return describe(make_runner(model, MAX_TOKENS_REASONING, params).run_trial(trial, num_stim))["thinking_tokens"]

    def result(mode, params, flagged, note):
        return {"usable": True, "reasoning_mode": mode, "params": json.dumps(params), "flagged": flagged, "note": note}

    try:
        default_tokens = thinking_tokens({})
    except errors.APIError as e:
        return {"usable": False, "reasoning_mode": "unusable", "params": None,
                "flagged": None, "note": f"{type(e).__name__}: {str(e)[:150]}"}
    note = f"default thinking tokens={default_tokens}"

    try:
        tokens = thinking_tokens(THINKING_OFF)
        if tokens == 0:
            return result("off", THINKING_OFF, False, f"{note}; off via thinking_budget=0")
        note += f"; thinking_budget=0 still thinks ({tokens})"
    except errors.ClientError:
        note += "; thinking_budget=0 rejected"

    for params in LOWEST_THINKING:
        setting = next(iter(params["thinking_config"].items()))
        try:
            tokens = thinking_tokens(params)
            return result(f"lowest={setting[0]}={setting[1]}", params, True, f"{note}; at lowest={tokens}")
        except errors.ClientError:
            note += f"; {setting[0]}={setting[1]} rejected"

    if default_tokens == 0:
        return result("non-thinking model", {}, False, note)
    return result("cannot be lowered", {}, True, note)


def runner_for(row: pd.Series) -> GoogleModelRunner:
    """Runner with the thinking setting and token limit chosen by the probe."""
    return make_runner(row["model"], MAX_TOKENS_REASONING if row["flagged"] else MAX_TOKENS, json.loads(row["params"]))


def plot(output_format: str = "svg"):
    """Plot performance against release date and save the summary table."""
    table = summarise(MODELS_CSV, OUTPUT_DIR)
    table["release_date"] = table["model"].map(RELEASE_DATES)
    missing = table.loc[table["release_date"].isna(), "model"].tolist()
    if missing:
        raise ValueError(f"Add release dates to RELEASE_DATES for: {missing}")
    plot_longitudinal(table, "Gemini", Path("output") / f"longitudinal_gemini.{output_format}")
    table.round(3).to_csv("output/longitudinal_gemini_summary.csv", index=False)
    print(table[["model", "release_date", "reasoning_mode", "accuracy", "rel_human"]].round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["probe", "run", "plot"])
    parser.add_argument("--workers", type=int, default=4, help="models processed in parallel")
    parser.add_argument("--models", nargs="*", help="run: only these models (default: all usable)")
    parser.add_argument("--skip-flagged", action="store_true", help="run: only models with thinking off")
    parser.add_argument("--format", default="svg", choices=["svg", "png", "pdf"], help="plot: output format")
    args = parser.parse_args()
    load_dotenv()

    if args.step == "probe":
        models = discover_models(genai.Client(api_key=os.getenv("GEMINI_KEY")))
        probe_models(models, partial(probe_model, loader=TaskLoader(task_paths()[0])), MODELS_CSV, args.workers)
    elif args.step == "run":
        run_models(MODELS_CSV, runner_for, describe, OUTPUT_DIR, args.workers, args.models, args.skip_flagged)
    else:
        plot(args.format)
