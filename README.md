# Neuropsych MiniBench

A comprehensive evaluation framework for testing vision-language models on neuropsychological tasks.

## Overview

This benchmark evaluates model performance across various visual-cognitive tasks inspired by neuropsychological assessments. Tasks span multiple difficulty stages and cognitive processes including visual recognition, spatial reasoning, and perceptual analysis.

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

# Load a task
loader = TaskLoader("test_specs/high/borb_triplet_shapes_meta.json")

# Run model evaluation
runner = OpenAIModelRunner(api_key="your-api-key", model_name="gpt-4o")
data = runner.generate_response(loader)

# Evaluate results
evaluator = Evaluator()
evaluator.evaluate(data)
results = evaluator.get_result()
```

## Project Structure

- **`loaders.py`** - Universal task loader for benchmark tasks
- **`runner.py`** - Model runners (OpenAI API integration)
- **`evaluator.py`** - Evaluation logic with task-specific scoring
- **`evaluator_config.json`** - Task categorization and evaluation methods
- **`naming_aliases.json`** - Shape and object name aliases for flexible matching
- **`datasets/`** - Task images organized by difficulty stage
- **`test_specs/`** - Task metadata and answer keys

## Task Categories

### Exact Match Tasks
- Orientation detection, size comparison, position analysis
- Requires precise string matching

### Naming Tasks  
- Object and shape recognition with alias support
- Handles alternative names (e.g., "rectangle" ↔ "square")

### Overlapping Tasks
- **Letters**: Character frequency matching
- **Shapes**: Shape frequency matching with aliases

### Triplet Tasks
- **Shapes**: Position-aware shape sequence matching
- Order matters, supports shape aliases

## Evaluation Features

- **Alias Support**: Flexible name matching via `naming_aliases.json`
- **Robust Parsing**: Handles malformed responses, None values
- **Task-Specific Logic**: Different scoring methods per task type
- **Detailed Logging**: Debug output for model vs. expected answers

## Configuration

Edit `evaluator_config.json` to:
- Add new tasks to evaluation categories
- Modify task groupings
- Configure evaluation methods

Edit `naming_aliases.json` to:
- Add shape/object name variations
- Support multilingual terms
- Handle common model output patterns

## Requirements

- Python 3.7+
- OpenAI API key for model evaluation
- Dependencies: `numpy`, `pandas`, `openai`, `tqdm`
