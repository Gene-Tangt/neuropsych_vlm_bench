from pathlib import Path

import pandas as pd

RESOURCES = Path(__file__).parent.parent / "resources"

STAGES = ["low", "mid", "high"]
STAGE_LABELS = {"low": "Low-level", "mid": "Mid-level", "high": "High-level"}
SUBTASK_GROUPING = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SUBTASK_GROUPING_LABELS = [
    "Simple Element Judgements",
    "Figure Judgements",
    "Occlusion and Overlap",
    "Property Biases",
    "Robustness Across Different Image cues",
    "Robustness Across Different Configurations",
    "Recognition in Visually Straightforward Circumstances",
    "View Invariance",
    "Semantic Association and Categorisation",
]

THREE_COLORS = ["#ff1f5b", "#ffc61e", "#009ade"]
NINE_COLORS = ["#ff1f5b", "#fef857", "#ffc61e", "#ff9205", "#bf602a", "#750f00", "#009ade", "#82d4bb", "#538f6c"]


def extract_model_name(path: str) -> str:
    """Model name from a results_<model>.csv path."""
    return Path(path).stem.removeprefix("results_")


def adjust_brightness(color, factor: float) -> tuple:
    """RGB colour scaled by factor, capped at 1."""
    return tuple(min(1, c * factor) for c in color)


def load_test_details() -> pd.DataFrame:
    """Per-test display name, subtask grouping, stage, full score and human norms."""
    columns = ["task", "display_name", "subtask_grouping", "stage", "full_score",
               "normative_score", "normative_SD", "normative_percent"]
    return pd.read_csv(RESOURCES / "test_details.csv")[columns]


def load_result_data(result_paths: list) -> dict:
    """Raw scores of each results CSV, keyed by model name."""
    return {extract_model_name(path): pd.read_csv(path)[["task", "raw_score"]].rename(
                columns={"raw_score": f"new_{extract_model_name(path)}_score"})
            for path in result_paths}


def merge_data(test_details: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Test details with every model's scores as extra columns."""
    data = test_details.copy()
    for result in results.values():
        data = data.merge(result, on="task", how="left")
    return data


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Add percent scores, performance relative to human and +-2 SD significance flags per model."""
    data["stage"] = pd.Categorical(data["stage"], categories=STAGES, ordered=True)
    data["subtask_grouping"] = pd.Categorical(
        data["subtask_grouping"], categories=SUBTASK_GROUPING, ordered=True).rename_categories(SUBTASK_GROUPING_LABELS)
    data = data.sort_values(["stage", "subtask_grouping"])

    score_columns = [col for col in data.columns if "_score" in col or "normative" in col]
    data[score_columns] = data[score_columns].apply(pd.to_numeric, errors="coerce")
    data["normative_percent"] = data["normative_score"] / data["full_score"] * 100
    data["normative_sd_percent"] = data["normative_SD"] / data["full_score"] * 100

    derived = {}
    for col in [c for c in data.columns if c.startswith("new_") and c.endswith("_score")]:
        prefix = col.removesuffix("_score")
        percent = data[col] / data["full_score"] * 100
        z = (data[col] - data["normative_score"]) / data["normative_SD"]
        derived[f"{prefix}_score_percent"] = percent
        derived[f"{prefix}_human"] = (percent - data["normative_percent"]) / data["normative_percent"]
        derived[f"{prefix}_sig_below"] = z <= -2
        derived[f"{prefix}_sig_above"] = z >= 2
    return pd.concat([data, pd.DataFrame(derived, index=data.index)], axis=1)
