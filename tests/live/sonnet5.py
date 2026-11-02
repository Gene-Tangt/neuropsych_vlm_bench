"""Shared setup for the live Claude Sonnet 5 regression run (before vs after the JSONL restructure).

Both runs use the same model settings as the "anthropic" entry of scripts/run_all.py, and
the same run loop, so the only thing that differs between them is the benchmark code.
Outputs go to results/regression/ (gitignored: unpublished results stay out of git).
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from neuropsych_bench import Evaluator, ModelConfig, create_runner

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "regression"

load_dotenv(ROOT / ".env")


def api_key():
    return os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")


# Same as scripts/run_all.py. Sonnet 5 rejects temperature, so it stays None.
# No refusal fallbacks: another model answering would contaminate the scores.
CONFIG = ModelConfig(model_name="claude-sonnet-5", api_key=api_key(), max_tokens=1000, temperature=None,
                     additional_params={"thinking": {"type": "disabled"}})


def trials_path(label: str) -> Path:
    return OUT_DIR / f"sonnet5_{label}_trials.csv"


def results_path(label: str) -> Path:
    return OUT_DIR / f"sonnet5_{label}_results.csv"


def run(label: str, tasks: list, make_loader) -> pd.DataFrame:
    """Run every task not already in the trials CSV, saving after each task; returns all trials.

    tasks are whatever make_loader takes (task file paths before the restructure, task names after).
    Each trial row also records the API stop_reason, so refusals and max_tokens cut-offs are visible.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = trials_path(label)
    done = pd.read_csv(path) if path.exists() else None
    done_tasks = set(done["task"]) if done is not None else set()
    runner = create_runner("anthropic", CONFIG)
    for i, task in enumerate(tasks):
        loader = make_loader(task)
        name = loader.get_task_info()["task"]
        if name in done_tasks:
            continue
        print(f"[{label}] task {i + 1}/{len(tasks)}: {name}", flush=True)
        info, trials = runner.generate_response(loader)
        evaluator = Evaluator()
        evaluator.evaluate((info, trials))
        rows = evaluator.get_trials().copy()
        rows["stop_reason"] = [t["api_response"].stop_reason for t in trials]
        done = rows if done is None else pd.concat([done, rows], ignore_index=True)
        done.to_csv(path, index=False)
    results = (done.groupby(["task", "task_type", "stage", "process"], sort=False)["correct"]
               .agg(num_trials="size", raw_score="sum").reset_index())
    results["percent_score"] = results["raw_score"] / results["num_trials"]
    results.to_csv(results_path(label), index=False)
    return done
