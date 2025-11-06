# Neuropsych VLM Benchmark Repo

This is the official repository for the paper "Visual Language Models show widespread visual deficits on neuropsychological tests" [[Link]](https://arxiv.org/abs/2504.10786v1).
The repository contains the code for running and evaluating the benchmarks for the open-source subset of tests (31 tests) used in the paper. Tests within this subset comprises of author's adaptation of some tests from Birmingham Object Recognition Battery (BORB), stimuli from the Leuven Embedded Figure Test (LEFT) [[FigShare Link]](https://figshare.com/articles/dataset/Leuven_Embedded_Figures_Test_Target_Shapes/3807885), and stimuli generated from the *MindSet: Vision* pipeline [[Paper]](https://arxiv.org/abs/2404.05290)[[Code]](https://github.com/ValerioB88/mindset-vision).

## Directory Structure

```
neuropsych_vlm_bench/
├── datasets/
├── test_specs/
├── utils/
├── normative_data_for_comparison/
├── run_all.py
├── demo_openai.ipynb
├── runner.py
├── evaluator.py
├── loaders.py
├── get_dataset.py
└── README.md
```
## Folders


* **`datasets/`** is a folder containing test images. The tests are categorized based on one of the three visual processes they tap into- low, mid, and high-level visual processes. Datasets images can be downloaded using the `get_dataset.py` script.
* **`test_specs/`** is a folder containing test specifications and configurations (Test metadata). Each test has a corresponding json file in this folder which contains information about the test such as the task type, stimuli path, the prompt, as well as the answer key. These metadata is used by the {} to run the tests.
* **`utils/`** is a folder containing utility files. For instance, it contains the naming aliases for the stimuli used in evaluation and information on the evaluation method of each test.
* **`normative_data_for_comparison/`** is a folder containing normative data for comparison. This subset of normative data is extracted from the normative data of the tests used in the paper [[OSF Link]](https://osf.io/ysxvg/overview).

## Core Files

* **`runner.py`** - VLM runner supporting OpenAI, Anthropic, and Google providers. Runner is a wrapper around the VLM providers and handles the API calls. This wrapper ensures that the API settings and prompt format are consistent with that used in the original paper.  
* **`loaders.py`** - This files contain the TaskLoader class that loads the task data from based on the task metadata in the test_specs folder. It also contains the function used to encode the stimuli images.
* **`evaluator.py`** - This file contains the Evaluator class that evaluates the model responses against the answer key stored on the task metadata in the test_specs folder.

* **`get_dataset.py`** - Dataset retrieval script. *See instructions below for downloading the dataset.*
* **`run_all.py`** - Main execution script for running the models used in the paper on the provided datasets.
* **`demo_openai.ipynb`** - Jupyter notebook demonstrating usage. *See end-to-end walkthrough here*

See `extra_info.txt` for more information on other files in the repository.


# Quick Start
## Set up the environment
**Python Virtual Environment**
```bash
python -m venv venv
pip install -r requirements.txt
```
**Conda Environment**
```bash
conda create -n neuropsych_vlm_bench python=3.12
conda activate neuropsych_vlm_bench
pip install -r requirements.txt
```
## Download the dataset using provided script
This will download the dataset to the `datasets` directory.

```bash
python get_dataset.py
```

## Run models used in the paper

The default script of run_all.py will run the models used in the paper on the provided datasets.
Please make sure to insert your API keys in `utils/api_keys.json` file.

```bash
python run_all.py
```
## Quickly testing on new models
The quickest way to run all tests on a new model is to modify the `run_all.py` file. Keep the providers that you want to test in the `run_all.py` file and remove the rest (if applicable). Then simply replace the model name in the `ModelConfig` with the new model name.

```python
openai_config = ModelConfig(
    model_name="<new_model_name>",
    api_key=api_keys["open_ai"],
    max_tokens=1000,
    temperature=1.0
)
```
The rest of the code remains the same.


# Detailed Walkthrough Can be Found Here
Demo of the pipeline with OpenAI API can be found in `demo_openai.ipynb`.

# Interactive Diagram
We also provide an AI-generatd (created with Windsurf), interactive diagram that might help users understand the pipeline better.
Access link: https://tinyurl.com/yc7y6ke2
