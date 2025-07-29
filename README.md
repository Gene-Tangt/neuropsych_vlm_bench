# VLM Neuropsych Minibench
VLMs evaluation framework based on neuropsychological and experimental psychology tasks.
Read more in: https://arxiv.org/abs/2504.10786v1

## Overview

This benchmark evaluates model performance across various visual-cognitive tasks based on neuropsychological assessments and psychological tasks.
Tasks used here are based on open-source datasets used in the paper. Neuropsychological tasks used here were re-rendered. 


## Project Structure

- **`loaders.py`** - Task loader to get payloads for model evaluation
- **`runner.py`** - Model runners (currently only OpenAI API integration. More to be added)
- **`evaluator.py`** - Evaluation logic with task-specific scoring
- **`evaluator_config.json`** - Task categorization based on their evaluation logic
- **`naming_aliases.json`** - Shape and object name aliases for flexible matching in naming tasks
- **`test_specs/`** - Task metadata and answer keys
- **`datasets/`** - Task images (* need to be downloaded separately from [here](https://drive.google.com/drive/folders/1qcAQBB9C1vf3PdaSPer4kNVBOrC7ORf4?usp=sharing) or run `get_dataset.py` to download) 


## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from loaders import TaskLoader
from runner import OpenAIModelRunner
from evaluator import Evaluator

# Evaluating a single task
loader = TaskLoader("test_specs/high/borb_triplet_shapes_meta.json") # get task payload
 
# Run model evaluation
runner = OpenAIModelRunner(api_key="your-api-key", model_name="gpt-4o")
data = runner.generate_response(loader) # generate model responses and append to the payload

# Evaluate results
evaluator = Evaluator()
evaluator.evaluate(data) # evaluate the results
results = evaluator.get_result() # returns a pandas DataFrame
results = evaluator.save_to_csv("results.csv") # saves the results to a CSV file
```
