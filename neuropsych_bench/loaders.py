import json


def task_paths(test_list: str = "test_specs/test_list.json") -> list:
    """Paths of every task metadata file, in stage order."""
    with open(test_list) as f:
        return [path for stage in json.load(f) for path in stage["task_paths"]]


class TaskLoader:
    """Task metadata and trials read from one test_specs/*_meta.json file."""

    def __init__(self, task_path: str):
        self.task_path = task_path
        with open(task_path) as f:
            self.task_data = json.load(f)

    def get_task_info(self) -> dict:
        """Stage, process, task name, task type and number of stimuli."""
        return {key: self.task_data.get(key) for key in ["stage", "process", "task", "task_type", "num_stim"]}

    def get_trials(self) -> list:
        """One dict per trial with the prompt, trial id, image paths and answer key."""
        return [{"prompt": self.task_data["prompt"], "trial_id": trial["trial"],
                 "images": trial["images"], "answer_key": trial["answer_key"]}
                for trial in self.task_data.get("trials", [])]
