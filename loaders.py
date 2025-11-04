"""Task Loader for Neuropsych Benchmark Tasks.

This module provides utilities for loading and parsing neuropsychological
benchmark task datasets from JSON files. The TaskLoader class handles reading
task metadata and trial information.

"""

import json
import base64
from typing import List, Dict

class TaskLoader:
    """Universal task loader for neuropsych benchmark tasks.
    
    Loads task configuration and trial data from JSON files. 
    Provides methods to access task metadata and formatted trial information.
    
    Attributes:
        task_path (str): Path to the task JSON file.
        task_data (Dict): Loaded task data including metadata and trials.
    
    Example:
        >>> loader = TaskLoader("tasks/visual_search.json")
        >>> info = loader.get_task_info()
        >>> print(info['task'], info['num_stim'])
    """

    def __init__(self, task_path: str):
        """Initialize the TaskLoader.
        
        Args:
            task_path (str): Path to the task JSON file.
        """
        self.task_path = task_path
        self.task_data = self._load_task()

    def _load_task(self) -> Dict:
        """Load the task JSON file.
        
        Returns:
            Dict: Parsed task data containing metadata and trials.
        """
        with open(self.task_path, 'r') as f:
            return json.load(f)

    def get_task_info(self) -> Dict:
        """Return metadata about the task.
        
        Extracts key task metadata fields including stage, process, task name,
        task type, and number of stimuli.
        
        Returns:
            Dict: Task metadata with keys:
                - stage (str): Visual processing state the test designed to tap into
                - process (str): Finer-grained process the test designed to tap into
                - task (str): Task name
                - task_type (str): Type of task
                - num_stim (str): Number of stimuli ('one', 'two', 'three', 'four')
        """
        return {
            key: self.task_data.get(key)
            for key in ["stage", "process", "task", "task_type", "num_stim"]
        }

    def get_trials(self, encode_image: bool = False) -> List[Dict]:
        """Return a list of formatted trial data.
        
        Extracts and formats trial information from the task data. Each trial
        includes the prompt, trial ID, image paths, and answer key.
        
        Args:
            encode_image (bool): Currently unused. Reserved for future functionality.
                Default: False.
        
        Returns:
            List[Dict]: List of trial dictionaries, each containing:
                - prompt (str): Task instruction text
                - trial_id (int): Trial identifier
                - images (Dict): Image paths organized by role (target, options, etc.)
                - answer_key (str): Correct answer for the trial
        """
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
        """Encode an image file as base64 string.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: Base64-encoded image string (UTF-8).
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
