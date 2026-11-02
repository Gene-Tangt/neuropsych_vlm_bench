import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from tqdm import tqdm

from ..prompting import build_parts


@dataclass
class ModelConfig:
    """Model name and request settings; temperature=None leaves it at the provider default."""
    model_name: str
    max_tokens: int = 100  # on reasoning models this also covers the reasoning tokens
    temperature: Optional[float] = 1.0
    api_key: Optional[str] = None
    additional_params: dict = field(default_factory=dict)  # provider-specific, e.g. reasoning_effort


def encode_image(path) -> str:
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

    def to_content(self, parts: list) -> list:
        """prompting.build_parts output in the provider's message format, in the same order."""
        return [self.text_part(p.text) if p.type == "text" else self.image_part(encode_image(p.path))
                for p in parts]

    def run_trial(self, trial: dict) -> Any:
        """Raw API response to one trial (a TaskLoader.get_trials item)."""
        return self.call_api(self.to_content(build_parts(trial, trial["data_root"])))

    def generate_response(self, loader) -> tuple:
        """Run every trial of a task; each trial gains 'api_response' and 'model_response'."""
        task_info, trials = loader.get_task_info(), loader.get_trials()
        for trial in tqdm(trials, desc="Getting model responses"):
            trial["api_response"] = self.run_trial(trial)
            trial["model_response"] = self.response_text(trial["api_response"])
        return task_info, trials
