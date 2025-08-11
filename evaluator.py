import numpy as np
import pandas as pd
import json
import re
from collections import Counter

with open("utils/evaluator_config.json", "r") as f:
    evaluator_config = json.load(f)

with open("utils/naming_aliases.json", "r") as f:
    naming_aliases = json.load(f)

class Evaluator:
    def __init__(self):

        self.result_table = pd.DataFrame(columns=[
            "task", "task_type", "stage", "process", "num_trials", "raw_score", "percent_score"
        ])

        self.exact_match_tasks = evaluator_config["exact_match_tasks"]
        self.naming_tasks = evaluator_config["naming_tasks"]
        self.overlapping_letters = evaluator_config["overlapping_letters"]
        self.overlapping_shapes = evaluator_config["overlapping_shapes"]
        self.borb_triplet_shapes = evaluator_config["borb_triplet_shapes"]
        self.aliases = naming_aliases

    def evaluate(self, data):

        if data[0]["task"] in self.exact_match_tasks:
            self._mark_exact(data)

        elif data[0]["task"] in self.naming_tasks:
            self._mark_naming(data)

        elif data[0]["task"] in self.overlapping_letters:
            self._mark_overlapping_letters(data)

        elif data[0]["task"] in self.overlapping_shapes:
            self._mark_overlapping_shapes(data)

        elif data[0]["task"] in self.borb_triplet_shapes:
            self._mark_borb_triplet_shapes(data)

        else:
            raise ValueError(f"Task {data[0]['task']} not found in evaluator config.")

    def get_result(self):

        return self.result_table

    def save_as_csv(self, path):

        if path is None:
            self.result_table.to_csv("results.csv", index=False)
            

        self.result_table.to_csv(path, index=False)



    def _mark_exact(self, data):

        task_score = 0

        for trial in data[1]:
            
            # Handle None responses
            if trial["model_response"] is None:
                continue
                
            brace_matches = re.findall(r"\{(.*?)\}", trial["model_response"])
            if not brace_matches or not brace_matches[0].strip():
                # No braces found or empty braces - mark as incorrect
                continue
                
            model_answer = brace_matches[0].strip().lower()

            if model_answer == trial["answer_key"]:
                task_score += 1

        # update result table

        self.result_table.loc[len(self.result_table)] = {
            "task": data[0]["task"],
            "task_type": data[0]["task_type"],
            "stage": data[0]["stage"],
            "process": data[0]["process"],
            "num_trials": len(data[1]),
            "raw_score": task_score,
            "percent_score": task_score / len(data[1])
        }

    def _mark_naming(self, data):

        task_score = 0 

        for trial in data[1]:

            # Handle None responses
            if trial["model_response"] is None:
                continue

            # Extract answer from braces, mark as incorrect if no braces or empty braces
            brace_matches = re.findall(r"\{(.*?)\}", trial["model_response"])
            if not brace_matches or not brace_matches[0].strip():
                # No braces found or empty braces - mark as incorrect
                continue
            
            model_answer = brace_matches[0].strip().lower()
            answer_key = trial["answer_key"].lower()
            
            # Check for exact match first
            if model_answer == answer_key:
                task_score += 1
            else:
                # Check aliases - find the aliases for the specific answer_key
                match_found = False
                
                # Check if answer_key exists in shape_aliases
                for shape, aliases in self.aliases["shape_aliases"].items():
                    if answer_key == shape.lower():
                        # Check exact match in aliases
                        if model_answer in [alias.lower() for alias in aliases]:
                            match_found = True
                            break
                        # Check if any alias appears as a word in the sentence
                        for alias in aliases:
                            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', model_answer):
                                match_found = True
                                break
                        if match_found:
                            break
                
                # Check if answer_key exists in object_aliases (only if not found in shapes)
                if not match_found:
                    for obj, aliases in self.aliases["object_aliases"].items():
                        if answer_key == obj.lower():
                            # Check exact match in aliases
                            if model_answer in [alias.lower() for alias in aliases]:
                                match_found = True
                                break
                            # Check if any alias appears as a word in the sentence
                            for alias in aliases:
                                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', model_answer):
                                    match_found = True
                                    break
                            if match_found:
                                break
                
                if match_found:
                    task_score += 1

        # update result table

        self.result_table.loc[len(self.result_table)] = {
            "task": data[0]["task"],
            "task_type": data[0]["task_type"],
            "stage": data[0]["stage"],
            "process": data[0]["process"],
            "num_trials": len(data[1]),
            "raw_score": task_score,
            "percent_score": task_score / len(data[1])
        }

    def _mark_overlapping_shapes(self, data):

        # Use already loaded shape aliases
        shape_aliases = self.aliases.get('shape_aliases', {})
        
        def normalize_shape(shape_name):
            """Normalize a shape name to its canonical form using aliases"""
            shape_lower = shape_name.strip().lower()
            for canonical, aliases in shape_aliases.items():
                if shape_lower in aliases:
                    return canonical
            return shape_lower

        task_score = 0 

        for trial in data[1]:

            # Handle None responses
            if trial["model_response"] is None:
                continue

            # Extract answer from braces, mark as incorrect if no braces or empty braces
            brace_matches = re.findall(r"\{(.*?)\}", trial["model_response"])
            if not brace_matches or not brace_matches[0].strip():
                # No braces found or empty braces - mark as incorrect
                continue

            # Parse model answer as comma-separated list
            model_answer_raw = brace_matches[0].strip().lower()
            model_shapes = [normalize_shape(shape) for shape in model_answer_raw.split(',')]


            if Counter(model_shapes) == Counter(trial['answer_key']):
                task_score += 1

        # update result table

        self.result_table.loc[len(self.result_table)] = {
            "task": data[0]["task"],
            "task_type": data[0]["task_type"],
            "stage": data[0]["stage"],
            "process": data[0]["process"],
            "num_trials": len(data[1]),
            "raw_score": task_score,
            "percent_score": task_score / len(data[1])
        }

    def _mark_borb_triplet_shapes(self, data):
        
        # Use already loaded shape aliases
        shape_aliases = self.aliases.get('shape_aliases', {})
        
        def normalize_shape(shape_name):
            """Normalize a shape name to its canonical form using aliases"""
            shape_lower = shape_name.strip().lower()
            for canonical, aliases in shape_aliases.items():
                if shape_lower in aliases:
                    return canonical
            return shape_lower

        task_score = 0 

        for trial in data[1]:

            # Handle None responses
            if trial["model_response"] is None:
                continue

            # Extract answer from braces, mark as incorrect if no braces or empty braces
            brace_matches = re.findall(r"\{(.*?)\}", trial["model_response"])
            if not brace_matches or not brace_matches[0].strip():
                # No braces found or empty braces - mark as incorrect
                continue

            # Parse model answer as comma-separated list
            model_answer_raw = brace_matches[0].strip().lower()
            model_shapes = [normalize_shape(shape) for shape in model_answer_raw.split(',')]
            
            # Normalize answer key shapes
            answer_key_shapes = [normalize_shape(shape) for shape in trial["answer_key"]]

            # Compare normalized lists (position matters)
            if model_shapes == answer_key_shapes:
                task_score += 1

        # update result table

        self.result_table.loc[len(self.result_table)] = {
            "task": data[0]["task"],
            "task_type": data[0]["task_type"],
            "stage": data[0]["stage"],
            "process": data[0]["process"],
            "num_trials": len(data[1]),
            "raw_score": task_score,
            "percent_score": task_score / len(data[1])
        }

    def _mark_overlapping_letters(self, data):

        task_score = 0 

        for trial in data[1]:

            # Handle None responses
            if trial["model_response"] is None:
                continue

            # Extract answer from braces, mark as incorrect if no braces or empty braces
            brace_matches = re.findall(r"\{(.*?)\}", trial["model_response"])
            if not brace_matches or not brace_matches[0].strip():
                # No braces found or empty braces - mark as incorrect
                continue

            # Parse model answer as comma-separated list
            model_answer = list(brace_matches[0].strip().lower())
            answer_key = list(trial["answer_key"].lower())

            if Counter(model_answer) == Counter(answer_key):
                task_score += 1

        # update result table

        self.result_table.loc[len(self.result_table)] = {
            "task": data[0]["task"],
            "task_type": data[0]["task_type"],
            "stage": data[0]["stage"],
            "process": data[0]["process"],
            "num_trials": len(data[1]),
            "raw_score": task_score,
            "percent_score": task_score / len(data[1])
        }
            

        


















