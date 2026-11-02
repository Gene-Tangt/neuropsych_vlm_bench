"""Run Claude Sonnet 5 on all 31 tasks with the pre-restructure code (about $7, ~1,130 API calls).

Writes results/regression/sonnet5_before_{trials,results}.csv, which
test_sonnet5_regression.py compares against. Re-running resumes from the last finished task.

Kept for provenance; it imports neuropsych_bench.loaders, which the restructure removed,
so it only runs on the git history from before it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sonnet5 import ROOT, api_key, run, trials_path  # noqa: E402

from neuropsych_bench.loaders import TaskLoader, task_paths  # noqa: E402

if __name__ == "__main__":
    if not api_key():
        sys.exit("Set ANTHROPIC_KEY or ANTHROPIC_API_KEY in .env")
    paths = [str(ROOT / p) for p in task_paths(str(ROOT / "test_specs/test_list.json"))]
    trials = run("before", paths, TaskLoader)
    print(f"{len(trials)} trials, accuracy {trials['correct'].mean():.3f} -> {trials_path('before')}")
