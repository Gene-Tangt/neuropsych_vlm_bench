"""Record what the pre-refactor runner sends to a model for every trial (no API calls).

Written once, against the code from before the JSONL restructure, to produce
baseline_content.jsonl: one line per trial with the ordered content parts, where
an image part is its file path instead of its base64 data. test_prompting.py checks
that the new prompting path reproduces it exactly.

Kept for provenance; it imports neuropsych_bench.runner and .loaders, which the
restructure removed, so it only runs on the git history from before it.
"""
import json
from pathlib import Path
from unittest import mock

from neuropsych_bench import loaders, runner

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("baseline_content.jsonl")


class RecordingRunner(runner.ModelRunner):
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


def main():
    rec = RecordingRunner(runner.ModelConfig(model_name="recording"))
    with mock.patch.object(runner, "encode_image", side_effect=lambda path: path.replace("\\", "/")):
        lines = []
        for path in loaders.task_paths(str(ROOT / "test_specs/test_list.json")):
            loader = loaders.TaskLoader(str(ROOT / path))
            info = loader.get_task_info()
            for trial in loader.get_trials():
                lines.append({"id": f"{info['task']}/{trial['trial_id']}",
                              "content": rec.run_trial(trial, info["num_stim"])})
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"{len(lines)} trials -> {OUT}")


if __name__ == "__main__":
    main()
