"""Run all tests on one model per provider and save results/results_<model>.csv and results/trials_<model>.csv.

API keys are read from .env (OPENAI_KEY, ANTHROPIC_KEY, GEMINI_KEY).
Usage: python scripts/run/run_all.py [openai] [anthropic] [google]   (default: all providers)
"""

import os
import sys
import time

from dotenv import load_dotenv

from neuropsych_bench import Evaluator, ModelConfig, TaskLoader, create_runner, task_names

load_dotenv()

# The paper used gpt-4o-2024-05-13 (temperature 1.0), claude-3-5-sonnet-20241022 (1.0) and gemini-1.5-pro (0.7),
# all now retired. Current models run with reasoning off (or at its minimum) and the provider's default temperature.
CONFIGS = {
    "openai": ModelConfig(
        model_name="gpt-6-sol", api_key=os.getenv("OPENAI_KEY"), max_tokens=1000, temperature=None,
        additional_params={"reasoning_effort": "none"}),
    "anthropic": ModelConfig(
        model_name="claude-sonnet-5", api_key=os.getenv("ANTHROPIC_KEY"), max_tokens=1000, temperature=None,
        additional_params={"thinking": {"type": "disabled"}}),
    "google": ModelConfig(
        # thinking cannot be disabled on Gemini 3.1 Pro; 1000 tokens truncated answers even at "low"
        model_name="gemini-3.1-pro-preview", api_key=os.getenv("GEMINI_KEY"), max_tokens=4000, temperature=None,
        additional_params={"thinking_config": {"thinking_level": "low"}}),
}

os.makedirs("results", exist_ok=True)
for provider in sys.argv[1:] or CONFIGS:
    runner = create_runner(provider, CONFIGS[provider])
    model = runner.config.model_name
    print(f"Testing: {model}")
    evaluator = Evaluator()
    tasks = task_names()
    for i, task in enumerate(tasks):
        print(f"\nProcessing task {i + 1}/{len(tasks)}: {task}")
        evaluator.evaluate(runner.generate_response(TaskLoader(task)))
        evaluator.save_trials_as_csv(f"results/trials_{model}.csv")  # saved per task so a crash keeps finished calls
        time.sleep(10)  # avoids rate limits on some APIs
    evaluator.save_as_csv(f"results/results_{model}.csv")
