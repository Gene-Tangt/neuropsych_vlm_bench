"""Longitudinal study of Claude models, following docs/longitudinal_inclusion_criteria.md.

    python scripts/longitudinal_claude.py probe   # discover models, test vision and whether thinking can be turned off
    python scripts/longitudinal_claude.py run     # run all tests on every usable model (resumable)
    python scripts/longitudinal_claude.py plot    # performance against release date

Outputs go to results/longitudinal_claude/. API key: ANTHROPIC_KEY in .env.
Refusal fallbacks are not enabled, as they would answer with a different model; refusals are counted instead.
"""

import argparse
import json
import os
import re
from functools import partial
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # per-trial progress bars are unreadable with parallel models

import anthropic
import pandas as pd
from dotenv import load_dotenv

from neuropsych_bench import AnthropicModelRunner, ModelConfig, TaskLoader, task_paths
from neuropsych_bench.longitudinal import plot_longitudinal, probe_models, run_models, summarise

OUTPUT_DIR = Path("results/longitudinal_claude")
MODELS_CSV = OUTPUT_DIR / "models.csv"

DATED = re.compile(r"-(\d{4})(\d{2})(\d{2})$")
# "disabled" is rejected by Claude Sonnet 5.5, where "between_tools" turns thinking off instead
THINKING_OFF = [{"type": "disabled"}, {"type": "between_tools"}]
LOWEST_EFFORT = {"output_config": {"effort": "low"}}

MAX_TOKENS = 1000
MAX_TOKENS_REASONING = 4000  # thinking tokens count toward the limit


def model_tier(model: str) -> str:
    """Model family; models of the same family are connected in plots."""
    return next((tier for tier in ("fable", "mythos", "opus", "sonnet", "haiku") if tier in model), "other")


def discover_models(client: anthropic.Anthropic) -> pd.DataFrame:
    """Claude models on the key, dropping undated aliases that have a dated snapshot."""
    models = {m.id: m.created_at for m in client.models.list() if m.id.startswith("claude-")}
    dated_bases = {DATED.sub("", m) for m in models if DATED.search(m)}
    rows = [{"model": m,
             "release_date": "-".join(DATED.search(m).groups()) if DATED.search(m) else created.strftime("%Y-%m-%d"),
             "tier": model_tier(m)}
            for m, created in models.items() if DATED.search(m) or m not in dated_bases]
    return pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)


def make_runner(model: str, max_tokens: int, params: dict) -> AnthropicModelRunner:
    """Runner at the provider's default temperature, with extra retries for long runs."""
    runner = AnthropicModelRunner(ModelConfig(model_name=model, api_key=os.getenv("ANTHROPIC_KEY"),
                                              max_tokens=max_tokens, temperature=None, additional_params=params))
    runner.client = runner.client.with_options(max_retries=8)
    return runner


def describe(response) -> dict:
    """Token usage, stop reason and whether the response contains thinking."""
    return {"input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,  # includes thinking tokens
            "thinking": any(block.type in ("thinking", "redacted_thinking") for block in response.content),
            "truncated": response.stop_reason == "max_tokens",
            "refusal": response.stop_reason == "refusal",
            "stop_reason": response.stop_reason}


def probe_model(model: str, loader: TaskLoader) -> dict:
    """Check the model answers a real trial, then whether thinking can be turned off or lowered."""
    trial, num_stim = loader.get_trials()[0], loader.get_task_info()["num_stim"]

    def thinks(params):
        return describe(make_runner(model, MAX_TOKENS_REASONING, params).run_trial(trial, num_stim))["thinking"]

    def result(mode, params, flagged, note):
        return {"usable": True, "reasoning_mode": mode, "params": json.dumps(params), "flagged": flagged, "note": note}

    try:
        default_thinks = thinks({})
    except anthropic.APIError as e:
        return {"usable": False, "reasoning_mode": "unusable", "params": None,
                "flagged": None, "note": f"{type(e).__name__}: {str(e)[:150]}"}
    note = f"default thinking={default_thinks}"

    # off only when explicitly disabled: adaptive thinking may skip thinking on a single trial
    for thinking in THINKING_OFF:
        try:
            if not thinks({"thinking": thinking}):
                return result("off", {"thinking": thinking}, False, f"{note}; off via {thinking['type']}")
            note += f"; {thinking['type']} still thinks"
        except anthropic.BadRequestError:
            note += f"; {thinking['type']} rejected"

    try:
        thinks(LOWEST_EFFORT)
        return result("lowest=low", LOWEST_EFFORT, True, note)
    except anthropic.BadRequestError:
        note += "; effort rejected"

    if not default_thinks:
        return result("non-thinking model", {}, False, note)
    return result("cannot be lowered", {}, True, note)


def runner_for(row: pd.Series) -> AnthropicModelRunner:
    """Runner with the thinking setting and token limit chosen by the probe."""
    return make_runner(row["model"], MAX_TOKENS_REASONING if row["flagged"] else MAX_TOKENS, json.loads(row["params"]))


def plot(output_format: str = "svg"):
    """Plot performance against release date and save the summary table."""
    table = summarise(MODELS_CSV, OUTPUT_DIR)
    plot_longitudinal(table, "Claude", Path("output") / f"longitudinal_claude.{output_format}")
    table.round(3).to_csv("output/longitudinal_claude_summary.csv", index=False)
    print(table[["model", "release_date", "tier", "reasoning_mode", "accuracy", "rel_human"]].round(3)
          .to_string(index=False))


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
        models = discover_models(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY")))
        probe_models(models, partial(probe_model, loader=TaskLoader(task_paths()[0])), MODELS_CSV, args.workers)
    elif args.step == "run":
        run_models(MODELS_CSV, runner_for, describe, OUTPUT_DIR, args.workers, args.models, args.skip_flagged)
    else:
        plot(args.format)
