"""Reads the benchmark from data/neuropsych_bench.jsonl (one row per trial; see data/README.md)."""
import json
from functools import lru_cache
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_JSONL = DATA_ROOT / "neuropsych_bench.jsonl"
TASK_FIELDS = ["stage", "process", "task", "task_type", "num_images"]


@lru_cache(maxsize=None)
def _load(path: str) -> tuple:
    with open(path, encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f if line.strip())


def load_rows(path=DEFAULT_JSONL) -> list:
    """Every trial row, in file order (low, mid, then high stage)."""
    return list(_load(str(Path(path).resolve())))


def task_names(path=DEFAULT_JSONL) -> list:
    """Name of every task, in file order."""
    return list(dict.fromkeys(row["task"] for row in load_rows(path)))


class TaskLoader:
    """Task metadata and trials of one task. Image paths in a row are relative to the JSONL's folder."""

    def __init__(self, task_name: str, path=DEFAULT_JSONL):
        self.task_name = task_name
        self.data_root = Path(path).resolve().parent
        self.rows = [row for row in load_rows(path) if row["task"] == task_name]
        if not self.rows:
            raise ValueError(f"Unknown task {task_name!r} in {path}")

    def get_task_info(self) -> dict:
        """Stage, process, task name, task type and number of images per trial."""
        return {key: self.rows[0][key] for key in TASK_FIELDS}

    def get_trials(self) -> list:
        """One dict per trial with the prompt, trial id, images, their roles and the answer key."""
        return [{"prompt": row["prompt"], "trial_id": row["id"].split("/", 1)[1], "images": row["images"],
                 "image_roles": row["image_roles"], "answer_key": row["answer"], "data_root": self.data_root}
                for row in self.rows]
