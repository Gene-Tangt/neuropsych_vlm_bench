# Neuropsych VLM Benchmark: data

The open-source subset of the benchmark from "Visual Language Models show widespread visual deficits on neuropsychological tests" ([arXiv:2504.10786](https://arxiv.org/abs/2504.10786v1)): 31 tests, 1,130 trials.

| File | Contents |
|---|---|
| `neuropsych_bench.jsonl` | One line per trial. This is the single source of truth: edit it directly. |
| `images/<stage>/<test>/…` | Test images (PNG). Paths in the JSONL are relative to this folder's parent (`data/`). |
| `normative_data.csv` | Human normative data for the 31 tests. |

This folder is self-contained. You can copy, zip or upload it without rewriting any paths.

## Row fields

```json
{"index": 242, "id": "left_embedded_figures/trial_000",
 "stage": "mid", "process": "figure_judgement", "task": "left_embedded_figures",
 "task_type": "match_to_sample", "num_images": 4,
 "prompt": "Below is a test. During the test you will see four images. ...",
 "images": ["images/mid/left_embedded_figures/trial_000/000.png", "…/000x.png", "…/000y.png", "…/000z.png"],
 "image_roles": ["target", "option_1", "option_2", "option_3"],
 "answer": "3"}
```

| Field | Meaning |
|---|---|
| `index` | Integer position in the file, 0 to 1129. |
| `id` | `<task>/<trial>`, unique. |
| `stage` | Visual processing stage the test taps: `low`, `mid` or `high`. Rows are ordered low → mid → high. |
| `process` | Finer-grained process within the stage. |
| `task` | Test name. Scoring rules per test are in `neuropsych_bench/resources/evaluator_config.json`. |
| `task_type` | `naming`, `yes_no`, `same_different` or `match_to_sample`. |
| `num_images` | Number of images in the trial (descriptive; equals `len(images)`). |
| `prompt` | Task instruction, the same for every trial of a test. The model is asked to put its final answer in `{}`. |
| `images` | Image paths **in presentation order**. |
| `image_roles` | Role of each image in `images`: `target`, `option_N` or `options`. |
| `answer` | Answer key. A string, except in `borb_overlapped_shapes` and `borb_triplet_shapes`, where it is a list of shape names. |

## Image layouts

| `image_roles` | Trials | Used by |
|---|---|---|
| `["target"]` | 760 | naming, yes/no, and `mindset_geometric_elements_comparison` (same/different with both shapes drawn in one image) |
| `["options", "options"]` | 140 | same/different pairs (line length, gap position, size, Weber's law) |
| `["target", "option_1", "option_2"]` | 166 | match-to-sample with two choices |
| `["target", "option_1", "option_2", "option_3"]` | 64 | match-to-sample with three choices (`left_embedded_figures`) |

## How a trial is presented

`neuropsych_bench/prompting.py` (`build_parts`, `PROMPT_VERSION`) defines what a model is sent for a trial:

1. the prompt;
2. each image in `images` order. In trials with a target plus numbered options, each image is preceded by a text cue: "Here's the target image", "Here's the first option", "Here's the second option", "Here's the third option".

Other harnesses should reproduce this exactly for their scores to be comparable.

## Sources and license

The tests are the authors' adaptations of tests from the Birmingham Object Recognition Battery (BORB), stimuli from the Leuven Embedded Figures Test (LEFT) ([FigShare](https://figshare.com/articles/dataset/Leuven_Embedded_Figures_Test_Target_Shapes/3807885)), and stimuli generated with the *MindSet: Vision* pipeline ([paper](https://arxiv.org/abs/2404.05290), [code](https://github.com/ValerioB88/mindset-vision)). The repository is released under the MIT License (see `LICENSE`).

## History

Until the restructure, the trials lived in 31 per-test files, `test_specs/<stage>/<test>_meta.json`, and the images in `datasets/`. `scripts/build/json_to_jsonl.py` converted them and checked that every row rebuilds its original trial. The old `num_stim` word ("one" to "four") became `num_images`.
