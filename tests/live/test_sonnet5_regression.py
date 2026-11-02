"""Live regression check: Claude Sonnet 5 on the restructured code vs the pre-restructure baseline.

Run with:  pytest -m live -s tests/live
Needs ANTHROPIC_KEY (or ANTHROPIC_API_KEY) and results/regression/sonnet5_before_trials.csv from
make_sonnet5_baseline.py. Costs about $7 (1,130 calls); re-running resumes the 'after' run.

Sonnet 5 has no temperature control, so two runs never match exactly. The test requires the
same trials and answer keys, a similar no-answer rate, and accuracy within sampling noise;
a broken prompt (missing images, wrong order or cues) shows up as a large drop in one of these.
"""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent))

from sonnet5 import OUT_DIR, api_key, run, trials_path  # noqa: E402

from neuropsych_bench import TaskLoader, task_names  # noqa: E402
from neuropsych_bench.scoring import extract_answer  # noqa: E402

pytestmark = pytest.mark.live

KEYS = ["task", "trial"]
META = ["task_type", "stage", "process"]
MAX_NO_ANSWER_RISE = 0.02  # share of trials
MAX_SE = 3  # overall and per-stage accuracy: allowed difference in standard errors
MAX_TASK_SHIFT = 0.40  # per task: only a structural break moves a task this far
NUM_TASKS = 31


def no_answer(trials: pd.DataFrame) -> pd.Series:
    """No {} answer could be extracted, or the model refused or hit max_tokens."""
    response = trials["model_response"].where(trials["model_response"].notna(), None)
    return response.map(lambda r: extract_answer(r) is None) | trials["stop_reason"].isin(["refusal", "max_tokens"])


def within_noise(before: pd.Series, after: pd.Series) -> tuple:
    p1, p2 = before.mean(), after.mean()
    se = math.sqrt(p1 * (1 - p1) / len(before) + p2 * (1 - p2) / len(after))
    return p2 - p1, se, abs(p2 - p1) <= MAX_SE * se if se else p1 == p2


@pytest.fixture(scope="module")
def runs():
    if not api_key():
        pytest.skip("ANTHROPIC_KEY / ANTHROPIC_API_KEY not set")
    if not trials_path("before").exists():
        pytest.fail(f"Baseline missing: run tests/live/make_sonnet5_baseline.py on the pre-restructure code "
                    f"to create {trials_path('before')}")
    before = pd.read_csv(trials_path("before"))
    after = run("after", task_names(), TaskLoader)
    return before, after


def test_same_trials(runs):
    before, after = runs
    assert len(before) == len(after) == 1130
    merged = before.merge(after, on=KEYS, how="outer", suffixes=("_before", "_after"), indicator=True)
    assert (merged["_merge"] == "both").all(), merged.loc[merged["_merge"] != "both", KEYS].head()
    for col in ["answer_key"] + META:
        diff = merged[merged[f"{col}_before"].astype(str) != merged[f"{col}_after"].astype(str)]
        assert diff.empty, f"{col} differs: {diff[KEYS].head().to_dict('records')}"


def test_no_answer_rate(runs):
    before, after = runs
    rate_before, rate_after = no_answer(before).mean(), no_answer(after).mean()
    print(f"\nno-answer rate: before {rate_before:.3f}, after {rate_after:.3f}")
    assert rate_after - rate_before <= MAX_NO_ANSWER_RISE


def test_accuracy(runs):
    before, after = runs
    rows = [("overall", *within_noise(before["correct"], after["correct"]))]
    for stage in ["low", "mid", "high"]:
        rows.append((stage, *within_noise(before.loc[before["stage"] == stage, "correct"],
                                          after.loc[after["stage"] == stage, "correct"])))

    tasks = []
    for task, b in before.groupby("task", sort=False):
        a = after[after["task"] == task]
        kb, ka = int(b["correct"].sum()), int(a["correct"].sum())
        p = fisher_exact([[kb, len(b) - kb], [ka, len(a) - ka]])[1]
        tasks.append({"task": task, "n": len(b), "before": kb / len(b), "after": ka / len(a),
                      "shift": ka / len(a) - kb / len(b), "p": p, "flagged": p < 0.05 / NUM_TASKS})
    tasks = pd.DataFrame(tasks)
    tasks.to_csv(OUT_DIR / "sonnet5_comparison.csv", index=False)

    agreement = (before.set_index(KEYS)["correct"] == after.set_index(KEYS)["correct"]).mean()
    print("\n" + tasks.round(3).to_string(index=False))
    for name, diff, se, ok in rows:
        print(f"{name:>8}: shift {diff:+.3f} (SE {se:.3f}){'' if ok else '  <-- outside noise'}")
    print(f"trial-level agreement (same correct/incorrect): {agreement:.3f}")
    print(f"flagged tasks (p < 0.05/{NUM_TASKS}): {tasks.loc[tasks['flagged'], 'task'].tolist()}")

    assert all(ok for *_, ok in rows), "overall or per-stage accuracy moved beyond sampling noise"
    big = tasks[tasks["shift"].abs() > MAX_TASK_SHIFT]
    assert big.empty, f"tasks shifted by more than {MAX_TASK_SHIFT:.0%}: {big['task'].tolist()}"
