"""Effect of reasoning effort on accuracy: Claude models on chosen tasks at chosen effort levels, same max_tokens.

    python scripts/run/thinking_level_effect.py                                   # claude-opus-5, borb_overlapped_letters, low/medium/high
    python scripts/run/thinking_level_effect.py --levels off low medium high      # with a thinking-off baseline
    python scripts/run/thinking_level_effect.py --models claude-opus-5 claude-sonnet-5 --tasks mindset_amodal_completion
    python scripts/run/thinking_level_effect.py --levels low --limit 3            # smoke test on the first 3 trials

Levels are adaptive thinking at output_config.effort low/medium/high/xhigh/max, or "off" (thinking disabled, no effort).
max_tokens covers thinking and answer alike and is the same for every level; truncated responses are logged.
The CoT is the summarized thinking the API returns (display "summarized"); the raw chain of thought is never exposed.
Refusal fallbacks are not enabled, as they would answer with a different model.

Outputs go to results/thinking_level_effect/: one trials_<model>_<task>_<level>_<max_tokens>.csv per run, and
summary.csv over all of them. Runs whose file exists are skipped; delete the file to redo one.
Runs with --limit go to results/thinking_level_effect/limited/ so they never stand in for a full run.
API key: ANTHROPIC_KEY in .env.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from neuropsych_bench import PROMPT_VERSION, AnthropicModelRunner, Evaluator, ModelConfig, TaskLoader, task_names

OUTPUT_DIR = Path("results/thinking_level_effect")

LEVELS = ["off", "low", "medium", "high", "xhigh", "max"]
MAX_TOKENS = 15000  # thinking tokens count toward the limit; below ~16k a non-streaming request stays within timeouts


def trials_path(output_dir: Path, model: str, task: str, level: str, max_tokens: int) -> Path:
    """Trial table of one run; its existence marks the run as done."""
    return output_dir / f"trials_{model}_{task}_{level}_{max_tokens}.csv"


def thinking_params(level: str) -> dict:
    """Request parameters for a level; "off" is only accepted at effort high or below, so it uses the default."""
    if level == "off":
        return {"thinking": {"type": "disabled"}}
    return {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": level}}


def make_runner(model: str, level: str, max_tokens: int) -> AnthropicModelRunner:
    """Runner at the provider's default temperature, with extra retries for long runs."""
    runner = AnthropicModelRunner(ModelConfig(model_name=model, api_key=os.getenv("ANTHROPIC_KEY"),
                                              max_tokens=max_tokens, temperature=None,
                                              additional_params=thinking_params(level)))
    runner.client = runner.client.with_options(max_retries=8)
    return runner


def describe(response) -> dict:
    """Token usage, stop reason and the summarized thinking (None if the model did not think)."""
    thoughts = [block.thinking for block in response.content if block.type == "thinking"]
    return {"input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,  # includes thinking tokens
            "thinking": any(block.type in ("thinking", "redacted_thinking") for block in response.content),
            "cot": "\n\n".join(thoughts) if any(thoughts) else None,
            "truncated": response.stop_reason == "max_tokens",
            "refusal": response.stop_reason == "refusal",
            "stop_reason": response.stop_reason}


class FirstTrials(TaskLoader):
    """TaskLoader restricted to the first `limit` trials (all of them when limit is None)."""

    def __init__(self, task_name: str, limit: int = None):
        super().__init__(task_name)
        self.limit = limit

    def get_trials(self) -> list:
        return super().get_trials()[:self.limit]


def run_level(model: str, level: str, task: str, max_tokens: int, limit: int = None) -> pd.DataFrame:
    """Trial table of one model on one task at one level."""
    info, trials = make_runner(model, level, max_tokens).generate_response(FirstTrials(task, limit))
    evaluator = Evaluator()
    evaluator.evaluate((info, trials))
    table = evaluator.get_trials()
    details = pd.DataFrame([describe(trial["api_response"]) for trial in trials])
    return pd.concat([pd.DataFrame({"model": model, "level": level, "max_tokens": max_tokens,
                                    "prompt_version": PROMPT_VERSION}, index=table.index), table, details], axis=1)


def summarise(trials: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, no-answer and truncation rates and output tokens per (model, level, task)."""
    order = {level: i for i, level in enumerate(LEVELS)}
    summary = (trials.assign(no_answer=trials["model_response"].isna())
               .groupby(["model", "level", "task", "stage", "max_tokens"], sort=False)
               .agg(num_trials=("correct", "size"), accuracy=("correct", "mean"), no_answer=("no_answer", "mean"),
                    thinking=("thinking", "mean"), truncated=("truncated", "mean"),
                    mean_output_tokens=("output_tokens", "mean"), median_output_tokens=("output_tokens", "median"))
               .reset_index())
    return summary.sort_values(["model", "task", "level"], key=lambda col: col.map(order) if col.name == "level"
                               else col, kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["claude-opus-5"])
    parser.add_argument("--tasks", nargs="+", default=["borb_overlapped_letters"], choices=task_names(),
                        metavar="TASK")
    parser.add_argument("--levels", nargs="+", default=["low", "medium", "high"], choices=LEVELS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--limit", type=int, help="only the first N trials of each task (for testing)")
    args = parser.parse_args()
    load_dotenv()

    output_dir = OUTPUT_DIR / "limited" if args.limit else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for task in args.tasks:
            for level in args.levels:
                path = trials_path(output_dir, model, task, level, args.max_tokens)
                if path.exists():
                    print(f"[{model}] {task} @ {level}: already done ({path.name}), skipped")
                    continue
                new = run_level(model, level, task, args.max_tokens, args.limit)
                new.to_csv(path, index=False)  # one file per run, so a crash keeps finished runs
                print(f"[{model}] {task} @ {level}: {int(new['correct'].sum())}/{len(new)} correct, "
                      f"{int(new['truncated'].sum())} truncated, {new['output_tokens'].mean():.0f} mean output tokens",
                      flush=True)

    trials = pd.concat([pd.read_csv(p) for p in sorted(output_dir.glob("trials_*.csv"))], ignore_index=True)
    summary = summarise(trials)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
