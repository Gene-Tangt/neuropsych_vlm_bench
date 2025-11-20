"""
This script runs all the tests on the models used in the paper.
If you would like to quickly test the models
- ensure you place your api keys in the utils/api_keys.json file.
- replace model names in the run_all.py file with the models you would like to test.
- (optional) remove the models that you don't want to test.
- Run the script.

The script will run all the tests on the models used in the paper and save the results in a csv file.
"""

import json
import os
from evaluator import Evaluator
from runner import ModelConfig, OpenAIModelRunner, AnthropicModelRunner, GoogleModelRunner
from loaders import TaskLoader
import time

# Get API keys
with open("utils/api_keys.json", "r") as f:
    api_keys = json.load(f)

# Load task paths
with open("test_specs/test_list.json", 'r') as file:
    all_task_paths = []
    for stage in json.load(file):
        all_task_paths.extend(stage['task_paths'])

# Setup models configuration

openai_config = ModelConfig(
    model_name="gpt-4o-2024-05-13",
    api_key=api_keys["open_ai"],
    max_tokens=1000,
    temperature=1.0
)

anthropic_config = ModelConfig(
    model_name="claude-3-5-sonnet-20241022",
    api_key=api_keys["anthropic"],
    max_tokens=1000,
    temperature=1.0
)

google_config = ModelConfig(
    model_name="gemini-1.5-pro",
    api_key=api_keys["google"],
    max_tokens=1000,
    temperature=0.7 # default at 0.7
)

# Initialize runners
openai_runner = OpenAIModelRunner(openai_config)
anthropic_runner = AnthropicModelRunner(anthropic_config)
google_runner = GoogleModelRunner(google_config)

# Initialize evaluator
batch_evaluator = Evaluator()

# Run all tests for each model # Remove the models that you don't want to test
for runner in [openai_runner, anthropic_runner, google_runner]:

    print(f"Testing: {runner.config.model_name}")

    # Loop through all tasks
    for i, task_path in enumerate(all_task_paths):
        print(f"\nProcessing task {i+1}/{len(all_task_paths)}: {task_path}")

        loader = TaskLoader(task_path) # Load task
        results = runner.generate_response(loader) # Generate response
        batch_evaluator.evaluate(results) # Evaluate response
        print(f"✓ Completed: {results[0]['task']}")
        time.sleep(10) # Timer was inserted here as a buffer to avoid rate limiting for some API
    
    os.makedirs("results", exist_ok=True)
    batch_evaluator.save_as_csv(f"results/results_{runner.config.model_name}.csv") # Save results
        
