import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from tqdm import tqdm

# Text cue placed before each image in 'three' and 'four' stimulus trials
IMAGE_CUES = ["Here's the target image", "Here's the first option", "Here's the second option",
              "Here's the third option"]


@dataclass
class ModelConfig:
    """Model name and request settings; temperature=None leaves it at the provider default."""
    model_name: str
    max_tokens: int = 100  # on reasoning models this also covers the reasoning tokens
    temperature: Optional[float] = 1.0
    api_key: Optional[str] = None
    additional_params: dict = field(default_factory=dict)  # provider-specific, e.g. reasoning_effort


def encode_image(path: str) -> str:
    """Base64 string of an image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class ModelRunner(ABC):
    """Runs the trials of a task on one model; subclasses implement one provider's API."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = self.create_client()

    @abstractmethod
    def create_client(self) -> Any:
        """Provider API client."""

    @abstractmethod
    def text_part(self, text: str) -> Any:
        """Text content in the provider's message format."""

    @abstractmethod
    def image_part(self, image: str) -> Any:
        """Base64 PNG content in the provider's message format."""

    @abstractmethod
    def call_api(self, content: list) -> Any:
        """Send one user message and return the raw API response."""

    @abstractmethod
    def response_text(self, response: Any) -> Optional[str]:
        """Answer text of a response, or None if there is none (scored incorrect)."""

    @staticmethod
    def prepare_images(trial: dict, num_stim: str) -> list:
        """Encoded images of a trial in presentation order."""
        images = trial["images"]
        if num_stim == "two":
            paths = images["options"]
        else:
            n_options = {"one": 0, "three": 2, "four": 3}[num_stim]
            paths = [images["target"][0]] + [images[f"option_{i}"][0] for i in range(1, n_options + 1)]
        return [encode_image(path) for path in paths]

    def format_conversation(self, instruction: str, images: list, num_stim: str) -> list:
        """Instruction followed by the images, each cued with text in three- and four-stimulus trials."""
        content = [self.text_part(instruction)]
        for i, image in enumerate(images):
            if num_stim in ("three", "four"):
                content.append(self.text_part(IMAGE_CUES[i]))
            content.append(self.image_part(image))
        return content

    def run_trial(self, trial: dict, num_stim: str) -> Any:
        """Raw API response to one trial."""
        return self.call_api(self.format_conversation(trial["prompt"], self.prepare_images(trial, num_stim), num_stim))

    def generate_response(self, loader) -> tuple:
        """Run every trial of a task; each trial gains 'api_response' and 'model_response'."""
        task_info, trials = loader.get_task_info(), loader.get_trials()
        for trial in tqdm(trials, desc="Getting model responses"):
            trial["api_response"] = self.run_trial(trial, task_info["num_stim"])
            trial["model_response"] = self.response_text(trial["api_response"])
        return task_info, trials


class OpenAIModelRunner(ModelRunner):
    def create_client(self):
        import openai
        return openai.OpenAI(api_key=self.config.api_key)

    def text_part(self, text):
        return {"type": "text", "text": text}

    def image_part(self, image):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}", "detail": "high"}}

    def call_api(self, content):
        # max_completion_tokens: reasoning models reject the deprecated max_tokens
        params = {"max_completion_tokens": self.config.max_tokens}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return self.client.chat.completions.create(
            model=self.config.model_name, messages=[{"role": "user", "content": content}],
            **params, **self.config.additional_params)

    def response_text(self, response):
        return response.choices[0].message.content


class AnthropicModelRunner(ModelRunner):
    def create_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self.config.api_key)

    def text_part(self, text):
        return {"type": "text", "text": text}

    def image_part(self, image):
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}}

    def call_api(self, content):
        # anthropic>=1.0 dropped the temperature argument; older models still accept it in the body
        params = {"extra_body": {"temperature": self.config.temperature}} if self.config.temperature is not None else {}
        return self.client.messages.create(
            model=self.config.model_name, max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": content}], **params, **self.config.additional_params)

    def response_text(self, response):
        # skips thinking blocks; a refusal has no text at all
        return "".join(block.text for block in response.content if block.type == "text") or None


class GoogleModelRunner(ModelRunner):
    def create_client(self):
        from google import genai
        from google.genai import types
        settings = {"max_output_tokens": self.config.max_tokens, "response_mime_type": "text/plain",
                    **self.config.additional_params}
        if self.config.temperature is not None:
            settings["temperature"] = self.config.temperature
        self.generation_config = types.GenerateContentConfig(**settings)
        return genai.Client(api_key=self.config.api_key)

    def text_part(self, text):
        return text

    def image_part(self, image):
        from google.genai import types
        return types.Part.from_bytes(data=base64.b64decode(image), mime_type="image/png")

    def call_api(self, content):
        return self.client.models.generate_content(
            model=self.config.model_name, contents=content, config=self.generation_config)

    def response_text(self, response):
        return response.text


RUNNERS = {"openai": OpenAIModelRunner, "anthropic": AnthropicModelRunner, "google": GoogleModelRunner}


def create_runner(provider: str, config: ModelConfig) -> ModelRunner:
    """Runner for 'openai', 'anthropic' or 'google'."""
    return RUNNERS[provider.lower()](config)
