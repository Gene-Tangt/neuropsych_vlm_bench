import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RESULT_COLUMNS = ["task", "task_type", "stage", "process", "num_trials", "raw_score", "percent_score"]
TRIAL_COLUMNS = ["task", "task_type", "stage", "process", "trial", "answer_key", "model_response", "correct"]

with open(DATA_DIR / "evaluator_config.json") as f:
    MARKING_METHOD = {task: method for method, tasks in json.load(f).items() for task in tasks}
with open(DATA_DIR / "naming_aliases.json") as f:
    ALIASES = json.load(f)


def extract_answer(response: Optional[str]) -> Optional[str]:
    """Lower-cased content of the first {...} in a response, or None if there is none or it is empty."""
    if response is None:
        return None
    matches = re.findall(r"\{(.*?)\}", response)
    if not matches or not matches[0].strip():
        return None
    return matches[0].strip().lower()


def canonical_shape(name: str) -> str:
    """Shape name mapped to its canonical form via the shape aliases."""
    name = name.strip().lower()
    return next((shape for shape, aliases in ALIASES["shape_aliases"].items() if name in aliases), name)


def matches_name(answer: str, answer_key: str) -> bool:
    """Answer equals the key, one of its aliases, or contains an alias as a whole word."""
    key = answer_key.lower()
    if answer == key:
        return True
    for group in ("shape_aliases", "object_aliases"):
        for name, aliases in ALIASES[group].items():
            if name.lower() == key and any(
                    answer == alias.lower() or re.search(r"\b" + re.escape(alias.lower()) + r"\b", answer)
                    for alias in aliases):
                return True
    return False


MARKERS = {
    "exact_match_tasks": lambda answer, key: answer == key,
    "naming_tasks": matches_name,
    # comma-separated shapes in any order
    "overlapping_shapes": lambda answer, key: Counter(map(canonical_shape, answer.split(","))) == Counter(key),
    # comma-separated shapes in the given order
    "borb_triplet_shapes": lambda answer, key: (list(map(canonical_shape, answer.split(",")))
                                                == list(map(canonical_shape, key))),
    # the same letters in any order
    "overlapping_letters": lambda answer, key: Counter(answer) == Counter(key.lower()),
}


class Evaluator:
    """Scores task responses, keeping per-task results and a trial-by-trial table."""

    def __init__(self):
        self.result_table = pd.DataFrame(columns=RESULT_COLUMNS)
        self.trial_table = pd.DataFrame(columns=TRIAL_COLUMNS)

    def evaluate(self, data: tuple):
        """Mark the (task_info, trials) output of ModelRunner.generate_response and add it to the tables."""
        info, trials = data
        if info["task"] not in MARKING_METHOD:
            raise ValueError(f"Task {info['task']} not found in evaluator config.")
        mark = MARKERS[MARKING_METHOD[info["task"]]]
        for trial in trials:
            answer = extract_answer(trial["model_response"])
            trial["correct"] = answer is not None and mark(answer, trial["answer_key"])

        score = sum(trial["correct"] for trial in trials)
        self.result_table.loc[len(self.result_table)] = {
            **{key: info[key] for key in ["task", "task_type", "stage", "process"]},
            "num_trials": len(trials), "raw_score": score, "percent_score": score / len(trials)}

        rows = pd.DataFrame([{
            **{key: info[key] for key in ["task", "task_type", "stage", "process"]},
            "trial": trial["trial_id"],
            "answer_key": trial["answer_key"] if isinstance(trial["answer_key"], str) else json.dumps(trial["answer_key"]),
            "model_response": trial["model_response"],
            "correct": trial["correct"],
        } for trial in trials], columns=TRIAL_COLUMNS)
        self.trial_table = rows if self.trial_table.empty else pd.concat([self.trial_table, rows], ignore_index=True)

    def get_result(self) -> pd.DataFrame:
        """Per-task results."""
        return self.result_table

    def get_trials(self) -> pd.DataFrame:
        """Trial-by-trial responses and correctness."""
        return self.trial_table

    def save_as_csv(self, path: str):
        """Save the per-task results."""
        self.result_table.to_csv(path, index=False)

    def save_trials_as_csv(self, path: str):
        """Save the trial-by-trial table."""
        self.trial_table.to_csv(path, index=False)
