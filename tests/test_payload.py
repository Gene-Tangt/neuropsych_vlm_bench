"""What each provider runner sends for a trial, checked against fake clients (no API calls)."""
import base64
from types import SimpleNamespace
from unittest import mock

import pytest
from google.genai import types

from neuropsych_bench import TaskLoader
from neuropsych_bench.models import (AnthropicModelRunner, GoogleModelRunner, ModelConfig, OpenAIModelRunner,
                                     create_runner)

CUED_TASK = "left_embedded_figures"  # target plus three numbered options


def make_runner(cls, **config):
    """Runner with a fake client, so nothing is sent."""
    with mock.patch.object(cls, "create_client", return_value=mock.MagicMock()):
        return cls(ModelConfig(model_name="m", **config))


def trial():
    return TaskLoader(CUED_TASK).get_trials()[0]


def image_bytes(t):
    return [(t["data_root"] / p).read_bytes() for p in t["images"]]


def expected_texts(t):
    return [t["prompt"], "Here's the target image", "Here's the first option", "Here's the second option",
            "Here's the third option"]


def test_create_runner():
    with mock.patch.object(AnthropicModelRunner, "create_client"):
        assert isinstance(create_runner("Anthropic", ModelConfig(model_name="m")), AnthropicModelRunner)
    with pytest.raises(KeyError):
        create_runner("other", ModelConfig(model_name="m"))


def test_anthropic_payload():
    runner, t = make_runner(AnthropicModelRunner, max_tokens=50, temperature=0.5), trial()
    runner.run_trial(t)
    kwargs = runner.client.messages.create.call_args.kwargs
    assert kwargs["model"] == "m" and kwargs["max_tokens"] == 50
    assert kwargs["extra_body"] == {"temperature": 0.5}
    [message] = kwargs["messages"]
    assert message["role"] == "user"
    content = message["content"]
    assert [p["type"] for p in content] == ["text"] + ["text", "image"] * 4
    assert [p["text"] for p in content if p["type"] == "text"] == expected_texts(t)
    images = [p["source"] for p in content if p["type"] == "image"]
    assert all(s["type"] == "base64" and s["media_type"] == "image/png" for s in images)
    assert [base64.b64decode(s["data"]) for s in images] == image_bytes(t)


def test_anthropic_default_temperature():
    runner = make_runner(AnthropicModelRunner, temperature=None)
    runner.run_trial(trial())
    assert "extra_body" not in runner.client.messages.create.call_args.kwargs


def test_openai_payload():
    runner, t = make_runner(OpenAIModelRunner, max_tokens=50, temperature=None,
                            additional_params={"reasoning_effort": "low"}), trial()
    runner.run_trial(t)
    kwargs = runner.client.chat.completions.create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 50 and "max_tokens" not in kwargs
    assert "temperature" not in kwargs and kwargs["reasoning_effort"] == "low"
    content = kwargs["messages"][0]["content"]
    assert [p["type"] for p in content] == ["text"] + ["text", "image_url"] * 4
    assert [p["text"] for p in content if p["type"] == "text"] == expected_texts(t)
    urls = [p["image_url"] for p in content if p["type"] == "image_url"]
    prefix = "data:image/png;base64,"
    assert all(u["url"].startswith(prefix) and u["detail"] == "high" for u in urls)
    assert [base64.b64decode(u["url"][len(prefix):]) for u in urls] == image_bytes(t)


def test_google_payload():
    with mock.patch("google.genai.Client"):
        runner = GoogleModelRunner(ModelConfig(model_name="m", max_tokens=50, temperature=0.5))
    assert runner.generation_config.max_output_tokens == 50 and runner.generation_config.temperature == 0.5
    t = trial()
    runner.run_trial(t)
    kwargs = runner.client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "m" and kwargs["config"] is runner.generation_config
    content = kwargs["contents"]
    assert [p for p in content if isinstance(p, str)] == expected_texts(t)
    images = [p for p in content if isinstance(p, types.Part)]
    assert len(content) == 9 and content[2::2] == images
    assert all(p.inline_data.mime_type == "image/png" for p in images)
    assert [p.inline_data.data for p in images] == image_bytes(t)


def test_response_text():
    anthropic = make_runner(AnthropicModelRunner)
    blocks = [SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text="{yes}")]
    assert anthropic.response_text(SimpleNamespace(content=blocks)) == "{yes}"
    assert anthropic.response_text(SimpleNamespace(content=blocks[:1])) is None

    openai = make_runner(OpenAIModelRunner)
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{no}"))])
    assert openai.response_text(reply) == "{no}"


def test_generate_response():
    runner = make_runner(AnthropicModelRunner)
    runner.client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text="{1}")])
    loader = TaskLoader(CUED_TASK)
    info, trials = runner.generate_response(loader)
    assert info == loader.get_task_info()
    assert len(trials) == runner.client.messages.create.call_count == len(loader.rows)
    assert all(t["model_response"] == "{1}" for t in trials)


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main(["-v", __file__]))
