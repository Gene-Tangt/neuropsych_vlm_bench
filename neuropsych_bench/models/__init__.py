from .anthropic_runner import AnthropicModelRunner
from .base import ModelConfig, ModelRunner, encode_image
from .google_runner import GoogleModelRunner
from .openai_runner import OpenAIModelRunner

RUNNERS = {"openai": OpenAIModelRunner, "anthropic": AnthropicModelRunner, "google": GoogleModelRunner}


def create_runner(provider: str, config: ModelConfig) -> ModelRunner:
    """Runner for 'openai', 'anthropic' or 'google'."""
    return RUNNERS[provider.lower()](config)


__all__ = ["AnthropicModelRunner", "GoogleModelRunner", "ModelConfig", "ModelRunner", "OpenAIModelRunner",
           "RUNNERS", "create_runner", "encode_image"]
