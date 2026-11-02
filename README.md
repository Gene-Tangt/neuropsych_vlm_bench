# Neuropsych VLM Benchmark

Code for running and scoring the open-source subset (31 tests) of the benchmark from "Visual Language Models show widespread visual deficits on neuropsychological tests" [[arXiv]](https://arxiv.org/abs/2504.10786v1).
The tests are the authors' adaptations of tests from the Birmingham Object Recognition Battery (BORB), stimuli from the Leuven Embedded Figures Test (LEFT) [[FigShare]](https://figshare.com/articles/dataset/Leuven_Embedded_Figures_Test_Target_Shapes/3807885), and stimuli generated with the *MindSet: Vision* pipeline [[Paper]](https://arxiv.org/abs/2404.05290) [[Code]](https://github.com/ValerioB88/mindset-vision).

The paper's data is on OSF [[Link]](https://osf.io/ysxvg/overview), including results replicated with this pipeline (`replication_with_pipeline`).

## Repository structure

```
neuropsych_vlm_bench/
├── data/                    the benchmark (see data/README.md)
│   ├── neuropsych_bench.jsonl   one line per trial: prompt, images, image roles, answer key
│   ├── images/                  test images, by stage and test
│   └── normative_data.csv       human normative data for the 31 tests
├── neuropsych_bench/        Python package
│   ├── dataset.py           TaskLoader: reads a test's trials from the JSONL
│   ├── prompting.py         builds what a model is sent for a trial (prompt, images, cues)
│   ├── models/              model runners for OpenAI, Anthropic and Google
│   ├── scoring.py           scores responses against the answer keys
│   ├── analysis/            human-norm comparison, plot palettes, resumable longitudinal runs
│   └── resources/           evaluator config, naming aliases, test details and norms
├── scripts/
│   ├── run/
│   │   ├── run_all.py                       run every test on one model per provider
│   │   ├── longitudinal_openai.py           every OpenAI model over release date
│   │   ├── longitudinal_claude.py           every Claude model over release date
│   │   └── longitudinal_gemini.py           flagship Gemini models over release date
│   ├── figures/
│   │   ├── generate_result_figures.py       paper figures (Supplementary Figures S2 and S3) and permutation tests
│   │   ├── generate_vendor_comparison.py    new models vs each vendor's previous model, per test
│   │   ├── generate_vendor_correlation.py   previous vs new accuracy per task, Spearman's rho
│   │   └── generate_vendor_longitudinal.py  vendor lines over release date
│   └── build/json_to_jsonl.py               how the JSONL was made from the old per-test JSON files
├── configs/                 model lists for the longitudinal figures
├── tests/                   pytest suite; tests/live/ calls paid APIs (pytest -m live)
├── examples/                paper results (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro)
├── notebooks/demo_openai.ipynb   walkthrough of the pipeline
├── docs/                    inclusion criteria for the longitudinal analysis
└── archive/                 shelved analyses (OpenAI error consistency), not maintained
```

Tests are grouped by the visual processing stage they tap into (low, mid or high level).

## Setup

```bash
conda create -n neuropsych_vlm_bench python=3.12
conda activate neuropsych_vlm_bench
pip install -e ".[dev]"   # dev adds pytest
pytest                     # offline checks of the data, prompts and scoring
```

The test images are in the repository (`data/images/`); there is nothing to download.

Copy `.env.example` to `.env` and add your API keys. Run all scripts from the repository root.

## Run models

```bash
python scripts/run/run_all.py                    # every provider
python scripts/run/run_all.py openai anthropic   # selected providers
```

To test another model, change its `ModelConfig` in `scripts/run/run_all.py`:

```python
ModelConfig(model_name="<model>", api_key=os.getenv("OPENAI_KEY"), max_tokens=1000, temperature=None,
            additional_params={"reasoning_effort": "none"})
```

`temperature=None` keeps the provider's default. On reasoning models `max_tokens` also covers the reasoning tokens; a response cut off before its answer is scored incorrect.

Each model produces two files in `results/`:
* `results_<model>.csv`: score per test.
* `trials_<model>.csv`: one row per trial with the raw response and whether it was marked correct.

## Using the package

```python
import os
from dotenv import load_dotenv
from neuropsych_bench import Evaluator, ModelConfig, TaskLoader, create_runner, task_names

load_dotenv()
runner = create_runner("anthropic", ModelConfig(model_name="claude-opus-5-5", api_key=os.getenv("ANTHROPIC_KEY"),
                                                max_tokens=1000, temperature=None))
evaluator = Evaluator()
for task in task_names():
    evaluator.evaluate(runner.generate_response(TaskLoader(task)))
evaluator.save_as_csv("results.csv")
```

Other providers are added by subclassing `ModelRunner`: implement `text_part`, `image_part`, `call_api` and `response_text`. What a model is sent for each trial comes from `neuropsych_bench.prompting.build_parts`, so every provider gets the same prompt, images and image cues.

## Data

The whole benchmark is `data/neuropsych_bench.jsonl` plus `data/images/`. Row fields, image layouts and sources are described in [`data/README.md`](data/README.md).

## Figures

Set the result CSVs at the bottom of `scripts/figures/generate_result_figures.py` and run it. The example CSVs reproduce the paper's Supplementary Figures S2 and S3. Figures are saved to `output/`.

## Longitudinal analysis

`scripts/run/longitudinal_openai.py`, `scripts/run/longitudinal_claude.py` and `scripts/run/longitudinal_gemini.py` (flagships only) run every model a provider still serves, following `docs/longitudinal_inclusion_criteria.md`:

```bash
python scripts/run/longitudinal_claude.py probe   # which models work, and how their reasoning can be controlled
python scripts/run/longitudinal_claude.py run     # all tests on every usable model; rerun to resume
python scripts/run/longitudinal_claude.py plot    # performance against release date, saved to output/
```

In the plots a solid dot means reasoning was turned off (or the model has none), and a dotted outline means reasoning could only be reduced to its lowest setting.

Results, trial-by-trial responses and token usage are saved to `results/longitudinal_<provider>/`.

## Interactive diagram

An AI-generated (Windsurf) interactive diagram of the pipeline: https://tinyurl.com/yc7y6ke2
