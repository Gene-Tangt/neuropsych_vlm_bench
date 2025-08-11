"""
Universal runner for the neuropsych minibench tasks.
Supports multiple VLM providers through a common interface.
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
    """Configuration for model parameters."""
    model_name: str
    max_tokens: int = 100
    temperature: float = 1.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}

class BaseModelRunner(ABC):
    """Abstract base class for VLM runners."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.task_info = None
        self.task = None
        
    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the API client for the specific provider."""
        pass
    
    @abstractmethod
    def _format_conversation(self, instruction: str, images: List[str]) -> Any:
        """Format the conversation for the specific provider's API."""
        pass
    
    @abstractmethod
    def _make_api_call(self, conversation: Any) -> str:
        """Make the API call and return the response text."""
        pass
    
    @staticmethod
    def encode_image_file(image_path: str):
        """Encode a local image file to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    
    def _prepare_images(self, trial: Dict) -> List[str]:
        """
        Prepare and encode images based on stimulus configuration.
        Stimulus configuration is stored in self.task_info, indicating the number of images used in a task

        Args:
            trial: A dictionary containing the trial information.

        Returns:
            List[str]: A list of base64 encoded image strings.
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
        
        Args:
            loader: TaskLoader object containing the task information and trials.
            Taskloader reads the task information and trials from a JSON file.
        
        Returns:
            tuple: A tuple containing the task information and the generated responses.
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
    """OpenAI API implementation of the model runner."""
    
    def _initialize_client(self) -> openai.OpenAI:
        """Initialize OpenAI client."""
        return openai.OpenAI(api_key=self.config.api_key)
    
    def _format_conversation(self, instruction: str, images: List[str]) -> List[Dict]:
        """Format conversation for OpenAI API."""
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
        """Make OpenAI API call."""
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
    """Anthropic Claude API implementation."""
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("anthropic package required for AnthropicModelRunner")
    
    def _format_conversation(self, instruction: str, images: List[str]) -> Dict:
        """Format conversation for Anthropic API."""
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
    
    def _make_api_call(self, conversation: Dict) -> str:
        """Make Anthropic API call."""
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
    """Google Gemini API implementation."""
    
    def _initialize_client(self):
        """Initialize Google client."""
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
        """Format conversation for Google API."""
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
        """Make Google API call."""
        if not hasattr(self, 'client'):
            self.client = self._initialize_client()
            
        # Generate content directly with the conversation list
        response = self.client.generate_content(conversation)
        
        return response.text


### ========================================= Factory Function ========================================= ###

def create_runner(provider: str, config: ModelConfig) -> BaseModelRunner:
    """Factory function to create model runners.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        config: ModelConfig object with provider settings
        
    Returns:
        BaseModelRunner: Appropriate runner instance
        
    Raises:
        ValueError: If provider is not supported
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
