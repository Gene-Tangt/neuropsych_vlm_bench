"""The new prompting path sends exactly what the pre-restructure runner sent, for every trial.

tests/fixtures/baseline_content.jsonl was recorded from the old runner (see make_baseline.py),
with image parts as their old repo-relative datasets/ paths instead of base64 data.
"""
import json
from pathlib import Path
from unittest import mock

import pytest

from neuropsych_bench import TaskLoader, task_names
from neuropsych_bench.models import ModelConfig, ModelRunner, base
from neuropsych_bench.prompting import build_parts, uses_cues

BASELINE = Path(__file__).parent / "fixtures" / "baseline_content.jsonl"


class RecordingRunner(ModelRunner):
    def create_client(self):
        return None

    def text_part(self, text):
        return ["text", text]

    def image_part(self, image):
        return ["image", image]

    def call_api(self, content):
        return content

    def response_text(self, response):
        return None


def old_path(path, data_root):
    """data/images/x -> datasets/x, the path the old runner encoded."""
    return "datasets/" + Path(path).relative_to(Path(data_root) / "images").as_posix()


def test_matches_old_runner():
    with open(BASELINE, encoding="utf-8") as f:
        baseline = {line["id"]: line["content"] for line in map(json.loads, f)}
    runner = RecordingRunner(ModelConfig(model_name="recording"))
    seen = 0
    for task in task_names():
        for trial in TaskLoader(task).get_trials():
            with mock.patch.object(base, "encode_image", side_effect=lambda p, root=trial["data_root"]: old_path(p, root)):
                content = runner.run_trial(trial)
            assert content == baseline[f"{task}/{trial['trial_id']}"], f"{task}/{trial['trial_id']}"
            seen += 1
    assert seen == len(baseline) == 1130


@pytest.mark.parametrize("roles, cued", [
    (["target"], False),
    (["options", "options"], False),
    (["target", "option_1", "option_2"], True),
    (["target", "option_1", "option_2", "option_3"], True),
])
def test_uses_cues(roles, cued):
    assert uses_cues(roles) is cued


def test_build_parts_order():
    row = {"prompt": "P", "images": ["a.png", "b.png", "c.png"], "image_roles": ["target", "option_1", "option_2"]}
    parts = build_parts(row, Path("root"))
    assert [p.type for p in parts] == ["text", "text", "image", "text", "image", "text", "image"]
    assert [p.path for p in parts if p.type == "image"] == [Path("root/a.png"), Path("root/b.png"), Path("root/c.png")]
