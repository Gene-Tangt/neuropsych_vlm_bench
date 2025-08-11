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

# Configs

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

# Runners

openai_runner = OpenAIModelRunner(openai_config)
anthropic_runner = AnthropicModelRunner(anthropic_config)
google_runner = GoogleModelRunner(google_config)

# Initialize the evaluator
batch_evaluator = Evaluator()

for runner in [openai_runner, anthropic_runner, google_runner]:

    print(f"Testing: {runner.config.model_name}")

    for i, task_path in enumerate(all_task_paths):
        print(f"\nProcessing task {i+1}/{len(all_task_paths)}: {task_path}")

        loader = TaskLoader(task_path)
        results = runner.generate_response(loader)
        batch_evaluator.evaluate(results)
        print(f"✓ Completed: {results[0]['task']}")
        time.sleep(10)
    
    batch_evaluator.save_as_csv(f"results_{runner.config.model_name}.csv")
        
