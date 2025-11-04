"""
Universal VLM Runner for Neuropsych Benchmark Tasks.

This module provides a unified interface for running neuropsychological benchmark
tasks across multiple Vision-Language Model (VLM) providers including OpenAI,
Anthropic Claude, and Google Gemini.

Architecture:
    - BaseModelRunner: Abstract base class using template method pattern
    - Provider-specific implementations: OpenAIModelRunner, AnthropicModelRunner, GoogleModelRunner
    - ModelConfig: Configuration dataclass for provider-agnostic settings
    - create_runner(): Factory function for instantiating runners

Example:
    >>> from runner import create_runner, ModelConfig
    >>> config = ModelConfig(model_name="gpt-4o", api_key="your-key")
    >>> runner = create_runner("openai", config)
    >>> results = runner.generate_response(task_loader)
"""

import os
import json
import base64
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import openai
from tqdm import tqdm

@dataclass
class ModelConfig:
    """Configuration for VLM model parameters.
    
    Attributes:
        model_name (str): Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro').
        max_tokens (int): Maximum tokens in model response. Default: 100.
        temperature (float): Sampling temperature (0.0-2.0). Higher = more random. Default: 1.0.
        api_key (Optional[str]): API key for authentication.
        additional_params (Dict[str, Any]): Provider-specific parameters (e.g., top_p, top_k).
    
    Example:
        >>> config = ModelConfig(
        ...     model_name="gpt-4o",
        ...     max_tokens=150,
        ...     temperature=0.7,
        ...     api_key=os.getenv("OPENAI_API_KEY")
        ... )
    """
    model_name: str
    max_tokens: int = 100
    temperature: float = 1.0
    api_key: Optional[str] = None
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}

class BaseModelRunner(ABC):
    """Blueprint for VLM runners using template method pattern.
    
    This class defines the common workflow for running neuropsych benchmark tasks
    across different VLM providers. Subclasses must implement provider-specific
    methods for client initialization, conversation formatting, and API calls.
    
    Template Method Pattern:
        generate_response() orchestrates the workflow:
        1. Load task info and trials
        2. For each trial: prepare images → format conversation → make API call
        3. Store responses in trial data
    
    Attributes:
        config (ModelConfig): Model configuration settings.
        task_info (Optional[Dict]): Task metadata (num_stim, task_name, etc.).
        task (Optional[List[Dict]]): List of trial dictionaries with prompts and images.
    
    Subclass Requirements:
        Must implement: _initialize_client(), _format_conversation(), _make_api_call()
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.task_info = None
        self.task = None
        
    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the API client for the specific provider.
        
        Returns:
            Any: Provider-specific client object (e.g., openai.OpenAI, anthropic.Anthropic).
        """
        pass
    
    @abstractmethod
    def _format_conversation(self, instruction: str, images: List[str]) -> Any:
        """Format the conversation for the specific provider's API.
        
        Converts task instruction and base64-encoded images into the provider's
        expected message format. Handles multi-image trials with text cues for
        'three' and 'four' stimulus configurations.
        
        Args:
            instruction (str): Task prompt/instruction text.
            images (List[str]): Base64-encoded image strings.
            
        Returns:
            Provider-specific conversation format.

        Note:
            Relies on self.task_info["num_stim"] to determine image layout.
        """
        pass
    
    @abstractmethod
    def _make_api_call(self, conversation: Any) -> str:
        """Make the API call and return the response text.
        
        Args:
            conversation (Any): Formatted conversation from _format_conversation().
            
        Returns:
            str: Model's text response content.
            
        """
        pass
    
    @staticmethod
    def encode_image_file(image_path: str) -> str:
        """Encode a local image file to base64 string.
        
        Args:
            image_path (str): Absolute or relative path to image file.
            
        Returns:
            str: Base64-encoded image string (UTF-8).
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    
    def _prepare_images(self, trial: Dict) -> List[str]:
        """Prepare and encode images based on stimulus configuration.
        
        Reads image paths from trial data and encodes them to base64 strings.
        The number and order of images depends on task_info["num_stim"]:
        
        - 'one': Single target image
        - 'two': Two option images
        - 'three': Target + 2 options
        - 'four': Target + 3 options
        
        Args:
            trial (Dict): Trial dictionary containing 'images' key with paths.
        
        Returns:
            List[str]: Base64-encoded image strings in presentation order.
            
        Note:
            Requires self.task_info to be set before calling.
        """
        images = []
        
        if self.task_info["num_stim"] == 'one':
            images.append(self.encode_image_file(trial["images"]["target"][0]))
            
        elif self.task_info["num_stim"] == 'two':
            for option in trial["images"]["options"]:
                images.append(self.encode_image_file(option))
                
        elif self.task_info["num_stim"] == 'three':
            images.append(self.encode_image_file(trial["images"]["target"][0]))
            images.append(self.encode_image_file(trial["images"]["option_1"][0]))
            images.append(self.encode_image_file(trial["images"]["option_2"][0]))
            
        elif self.task_info["num_stim"] == 'four':
            images.append(self.encode_image_file(trial["images"]["target"][0]))
            images.append(self.encode_image_file(trial["images"]["option_1"][0]))
            images.append(self.encode_image_file(trial["images"]["option_2"][0]))
            images.append(self.encode_image_file(trial["images"]["option_3"][0]))
            
        return images
    
    def generate_response(self, loader) -> tuple:
        """Generate responses using the configured VLM provider.
        
        Main entry point for running benchmark tasks. Iterates through all trials,
        prepares images, formats conversations, and collects model responses.
        
        Args:
            loader: TaskLoader object containing the task information and trials.
                TaskLoader reads the task information and trials from a JSON file.
        
        Returns:
            tuple: A tuple containing:
                - task_info (Dict): Task metadata including num_stim, task_name, etc.
                - task (List[Dict]): List of trials with added 'conversation' and 'model_response' fields.
        """
        self.task_info = loader.get_task_info()
        self.task = loader.get_trials()
        
        for trial in tqdm(self.task, desc="Getting model responses"):
            instruction = trial["prompt"]
            images = self._prepare_images(trial) # encode stimuli images
            
            conversation = self._format_conversation(instruction, images)
            response_text = self._make_api_call(conversation)
            
            trial["conversation"] = conversation
            trial["model_response"] = response_text
            
        return (self.task_info, self.task)

### ========================================= OpenAI Model Runner ========================================= ###

class OpenAIModelRunner(BaseModelRunner):
    """OpenAI API implementation of the model runner.
    
    Supports OpenAI models (GPT-4, GPT-4o, GPT-5, etc.) and OpenAI-compatible APIs.
    Uses the official openai Python package.
    
    API Format:
        - Messages: List of dicts with 'role' and 'content'
        - Images: Embedded as base64 data URLs with detail level
        
    Note:
        Requires OPENAI_API_KEY environment variable or config.api_key.
    """
    
    def _initialize_client(self) -> openai.OpenAI:
        """Initialize OpenAI client."""
        return openai.OpenAI(api_key=self.config.api_key)
    
    def _format_conversation(self, instruction: str, images: List[str]) -> List[Dict]:
        """Format conversation for OpenAI API.
        
        Args:
            instruction (str): Task instruction text.
            images (List[str]): Base64-encoded image strings.
            
        """
        content = [{"type": "text", "text": instruction}] # always start with the instruction
        
        # add text cues for multi-image trials
        if self.task_info["num_stim"] == 'three':
            content.append({"type": "text", "text": "Here's the target image"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[0]}", "detail": "high"}})
            content.append({"type": "text", "text": "Here's the first option"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[1]}", "detail": "high"}})
            content.append({"type": "text", "text": "Here's the second option"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[2]}", "detail": "high"}})
            
        elif self.task_info["num_stim"] == 'four':
            content.append({"type": "text", "text": "Here's the target image"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[0]}", "detail": "high"}})
            content.append({"type": "text", "text": "Here's the first option"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[1]}", "detail": "high"}})
            content.append({"type": "text", "text": "Here's the second option"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[2]}", "detail": "high"}})
            content.append({"type": "text", "text": "Here's the third option"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[3]}", "detail": "high"}})
            
        else:
            # For 'one' and 'two' stimulus cases: these contain just images no text cues
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}", "detail": "high"}})
        
        return [{"role": "user", "content": content}]
    
    def _make_api_call(self, conversation: List[Dict]) -> str:
        """Make OpenAI API call.
        
        Args:
            conversation (List[Dict]): Formatted OpenAI messages.
            
        Returns:
            str: Model response text content.
        """
        if not hasattr(self, 'client'):
            self.client = self._initialize_client()
            
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=conversation,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            **self.config.additional_params
        )
        
        return response.choices[0].message.content

### ========================================= Anthropic Model Runner ========================================= ###

class AnthropicModelRunner(BaseModelRunner):
    """Anthropic Claude API implementation.
    
    Supports Claude models (Claude Opus, Sonnet, Haiku, etc.).
    Requires the anthropic Python package.
    
    API Format:
        - Messages: List of dicts with 'role' and 'content'
        - Images: Base64 data with explicit media_type
        
    Installation:
        pip install anthropic
        
    Note:
        Requires ANTHROPIC_API_KEY environment variable.
    """
    
    def _initialize_client(self):
        """Initialize Anthropic client.
        
        Returns:
            anthropic.Anthropic: Configured Anthropic client instance.
            
        Raises:
            ImportError: If anthropic package is not installed.
        """
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("anthropic package required for AnthropicModelRunner")
    
    def _format_conversation(self, instruction: str, images: List[str]) -> List[Dict]:
        """Format conversation for Anthropic API.
        
        Args:
            instruction (str): Task instruction text.
            images (List[str]): Base64-encoded image strings.
            
        Returns:
            List[Dict]: Anthropic message format with role='user' and multimodal content.
        """
        content = [{"type": "text", "text": instruction}] # always start with the instruction
        
        # add text cues for multi-image trials
        if self.task_info["num_stim"] == 'three':
            content.append({"type": "text", "text": "Here's the target image"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[0]}})
            content.append({"type": "text", "text": "Here's the first option"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[1]}})
            content.append({"type": "text", "text": "Here's the second option"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[2]}})
            
        elif self.task_info["num_stim"] == 'four':
            content.append({"type": "text", "text": "Here's the target image"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[0]}})
            content.append({"type": "text", "text": "Here's the first option"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[1]}})
            content.append({"type": "text", "text": "Here's the second option"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[2]}})
            content.append({"type": "text", "text": "Here's the third option"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": images[3]}})
            
        else:
            # For 'one' and 'two' stimulus cases: these contain just images no text cues
            for image in images:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}})
        
        return [{"role": "user", "content": content}]
    
    def _make_api_call(self, conversation: List[Dict]) -> str:
        """Make Anthropic API call.
        
        Args:
            conversation (List[Dict]): Formatted Anthropic messages.
            
        Returns:
            str: Model response text content.
        """
        if not hasattr(self, 'client'):
            self.client = self._initialize_client()
            
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages= conversation,
            **self.config.additional_params
        )
        
        return response.content[0].text

### ========================================= Google Model Runner ========================================= ###

class GoogleModelRunner(BaseModelRunner):
    """Google Gemini API implementation.
    
    Supports Gemini models (Gemini 1.5 Pro, Flash, etc.).
    Requires the google-generativeai Python package.
    
    API Format:
        - Content: Flat list of alternating text strings and image dicts
        - Images: Decoded base64 bytes with mime_type
        
    Installation:
        pip install google-generativeai
        
    Note:
        Requires GOOGLE_API_KEY environment variable or config.api_key.
    """
    
    def _initialize_client(self):
        """Initialize Google Gemini client.
        
        Returns:
            genai.GenerativeModel: Configured Gemini model instance.
            
        Raises:
            ImportError: If google-generativeai package is not installed.
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "response_mime_type": "text/plain",
                **self.config.additional_params
            }
            
            return genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config
            )
        except ImportError:
            raise ImportError("google-generativeai package required for GoogleModelRunner")
    
    def _format_conversation(self, instruction: str, images: List[str]) -> List:
        """Format conversation for Google Gemini API.
        
        Args:
            instruction (str): Task instruction text.
            images (List[str]): Base64-encoded image strings.
            
        Returns:
            List: Flat list of text strings and image dicts for Gemini API.
        """
        content = [instruction] # always start with the instruction
        
        # Convert base64 strings to proper format for Google API
        encoded_images = []
        for img_b64 in images:
            encoded_images.append({
                'mime_type': 'image/png',
                'data': base64.b64decode(img_b64)
            })
        
        # add text cues for multi-image trials
        if self.task_info["num_stim"] == 'three':
            content.append("Here's the target image")
            content.append(encoded_images[0])
            content.append("Here's the first option")
            content.append(encoded_images[1])
            content.append("Here's the second option")
            content.append(encoded_images[2])
            
        elif self.task_info["num_stim"] == 'four':
            content.append("Here's the target image")
            content.append(encoded_images[0])
            content.append("Here's the first option")
            content.append(encoded_images[1])
            content.append("Here's the second option")
            content.append(encoded_images[2])
            content.append("Here's the third option")
            content.append(encoded_images[3])
            
        else:
            # For 'one' and 'two' stimulus cases: these contain just images no text cues
            for image in encoded_images:
                content.append(image)
        
        return content
    
    def _make_api_call(self, conversation: List) -> str:
        """Make Google Gemini API call.
        
        Args:
            conversation (List): Formatted Gemini content list.
            
        Returns:
            str: Model response text content.
        """
        if not hasattr(self, 'client'):
            self.client = self._initialize_client()
            
        # Generate content directly with the conversation list
        response = self.client.generate_content(conversation)
        
        return response.text


### ========================================= Factory Function ========================================= ###

def create_runner(provider: str, config: ModelConfig) -> BaseModelRunner:
    """Factory function to create model runners.
    
    Args:
        provider (str): Provider name ('openai', 'anthropic', 'google').
            Case-insensitive.
        config (ModelConfig): ModelConfig object with provider settings.
        
    Returns:
        BaseModelRunner: Appropriate runner instance for the provider.
        
    Raises:
        ValueError: If provider is not supported.
        
    Example:
        >>> config = ModelConfig(model_name="gpt-4o", api_key="sk-...")
        >>> runner = create_runner("openai", config)
        >>> # Or for Anthropic:
        >>> runner = create_runner("anthropic", ModelConfig(
        ...     model_name="claude-3-5-sonnet-20241022",
        ...     api_key="sk-ant-..."
        ... ))
    """
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIModelRunner(config)
    elif provider == "anthropic":
        return AnthropicModelRunner(config)
    elif provider == "google":
        return GoogleModelRunner(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, anthropic, google")
