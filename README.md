## Core Files

* **`runner.py`** - VLM runner supporting OpenAI, Anthropic, and Google providers
* **`evaluator.py`** - Scoring system
* **`loaders.py`** - Task data loading functions
* **`get_dataset.py`** - Dataset retrieval script
* **`run_all.py`** - Main execution script for running benchmarks
* **`demo_openai.ipynb`** - Jupyter notebook demonstrating usage

## Directories

* **`datasets/`** - Task datasets
* **`test_specs/`** - Test specifications and configurations (Test metadata)
* **`utils/`** - Configuration files
* **`normative_data_for_comparison/`** - Reference data for comparison

## Setup

```bash
python -m venv venv
pip install -r requirements.txt
```
## Get Dataset
This will download the dataset to the `datasets` directory.

```bash
python get_dataset.py
```

## Run models used in the paper

Make sure to set the API keys in a .json file.

```bash
python run_all.py
```

## Demo
Demo the usage of the runner with OpenAI can be found in `demo_openai.ipynb`.

