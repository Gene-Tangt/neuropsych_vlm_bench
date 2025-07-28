import json
import base64
from typing import List, Dict

class TaskLoader:
    """Simplified universal task loader for neuropsych benchmark tasks."""

    def __init__(self, task_path: str):
        self.task_path = task_path
        self.task_data = self._load_task()

    def _load_task(self) -> Dict:
        """Load the task JSON file."""
        with open(self.task_path, 'r') as f:
            return json.load(f)

    def get_task_info(self) -> Dict:
        """Return metadata about the task."""
        return {
            key: self.task_data.get(key)
            for key in ["stage", "process", "task", "task_type", "num_stim"]
        }

    def get_trials(self, encode_image: bool = False) -> List[Dict]:
        """Return a list of formatted trial data."""
        trials = []

        for trial in self.task_data.get("trials", []):
            trials.append({
                "prompt": self.task_data["prompt"],
                "trial_id": trial["trial"],
                "images": trial["images"],
                "answer_key": trial["answer_key"],
            })
        
        return trials

    @staticmethod
    def encode_image_file(image_path: str) -> str:
        """Encode an image file as base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
