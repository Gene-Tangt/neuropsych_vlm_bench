# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Benchmark harness for running vision-language models on 31 neuropsychological vision tests (1,130 trials) from the paper "Visual Language Models show widespread visual deficits on neuropsychological tests" (arXiv:2504.10786). Tests come from BORB, LEFT and MindSet: Vision, grouped by visual processing stage (`low`, `mid`, `high`).

## Commands

```bash
pip install -e ".[dev]"                       # package + pytest (Python >=3.10; README uses a conda env on 3.12)
pytest                                        # offline unit tests, no API calls
pytest tests/test_scoring.py::test_marking    # single test
python tests/test_dataset.py                  # one file on its own, verbose PASSED/FAILED per test

python scripts/run/run_all.py [openai] [anthropic] [google]   # every test on one model per provider
python scripts/run/longitudinal_claude.py probe|run|plot      # also _openai.py, _gemini.py
python scripts/run/thinking_level_effect.py --levels off low --limit 3   # Claude accuracy vs reasoning effort; --limit = smoke test
python scripts/model_info_sheet.py [anthropic] [openai] [google]       # dump available models + metadata to docs/
python scripts/figures/generate_result_figures.py             # paper figures; set RESULT_CSVS at bottom of the file
python scripts/figures/generate_vendor_comparison.py          # also _correlation.py; set VENDOR_PAIRS at bottom
python scripts/figures/generate_vendor_longitudinal.py        # reads results/longitudinal_*/ via configs/main_line_models.json
```

There is no linter or formatter configured. Run all scripts from the repository root: outputs go to relative `results/` and `output/` (both gitignored). `examples/` holds the paper's original result CSVs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro; all retired models) and is the default input to the figure scripts. API keys live in `.env` as `OPENAI_KEY`, `ANTHROPIC_KEY`, `GEMINI_KEY` (see `.env.example`).

## Architecture

The pipeline is: **JSONL row → `TaskLoader` → `build_parts` → provider runner → `Evaluator`**.

- **Data is a single source of truth**: `data/neuropsych_bench.jsonl`, one row per trial, edited directly. Image paths are relative to `data/`. `answer` is a string except in `borb_overlapped_shapes` and `borb_triplet_shapes` (list of shape names). Row schema and image layouts are documented in `data/README.md`. The old per-test `test_specs/*_meta.json` + `datasets/` layout no longer exists; `scripts/build/json_to_jsonl.py` is kept only for provenance.
- **`neuropsych_bench/dataset.py`**: `load_rows` (cached), `task_names` (file order low→mid→high), `TaskLoader(task)` with `get_task_info()` / `get_trials()`.
- **`neuropsych_bench/prompting.py`** is the *only* place that decides what a model sees: prompt text, then images in order, each preceded by a role cue ("Here's the target image", "Here's the first option", …) only when the trial has a target plus numbered options. Runners must not reorder, add or drop parts. Bump `PROMPT_VERSION` whenever `build_parts` output changes.
- **`neuropsych_bench/models/`**: `ModelRunner` (in `base.py`) owns the trial loop (`generate_response`) and image encoding; providers subclass it and implement only `create_client`, `text_part`, `image_part`, `call_api`, `response_text`. `create_runner(provider, ModelConfig)` picks one of `openai`/`anthropic`/`google`. `ModelConfig.temperature=None` means provider default; `max_tokens` includes reasoning tokens; provider-specific options (reasoning effort, thinking config) go in `additional_params`. `response_text` returns `None` for no answer (refusal, thinking-only), which scores incorrect.
- **`neuropsych_bench/scoring.py`**: the answer is the lower-cased content of the *first* `{...}` in the response. The marking method per task comes from `neuropsych_bench/resources/evaluator_config.json` (shipped as package data; the `resources/` paths below are relative to `neuropsych_bench/`) (`exact_match_tasks`, `naming_tasks`, `overlapping_shapes`, `borb_triplet_shapes`, `overlapping_letters`); naming uses whole-word alias matching from `resources/naming_aliases.json`. A new task must be added to `evaluator_config.json` or `Evaluator.evaluate` raises.
- **`neuropsych_bench/analysis/`**: human-norm comparison helpers and palettes (`__init__.py`, using `resources/test_details.csv`), and `longitudinal.py`, the shared probe/run/plot machinery. Runs are resumable: `run_model` skips tasks already in `results_<model>.csv` and appends to the responses log. The per-provider `scripts/run/longitudinal_*.py` handle model discovery and how to turn reasoning off, following `docs/longitudinal_inclusion_criteria.md` (reasoning off ⇒ 0 reasoning tokens, else lowest setting and flagged; 1,000 max tokens, 4,000 for flagged models; default sampling). `configs/main_line_models.json` lists models per vendor line for the vendor figures.

## Tests

Unit tests only, all offline: `test_dataset.py` (JSONL rows, task order, `TaskLoader`), `test_images.py` (every image exists, is a valid PNG, base64 round-trips), `test_prompting.py` (`build_parts` order and cues for every trial), `test_payload.py` (each provider's request built from a trial, against mocked clients) and `test_scoring.py` (answer extraction and marking methods).

`archive/` is shelved and not maintained.
