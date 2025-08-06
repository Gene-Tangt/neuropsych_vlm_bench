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
    temperature: float = 0.0
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
    def encode_image_file(image_path: str) -> str:
        """Encode a local image file to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    
    def _prepare_images(self, trial: Dict) -> List[str]:
        """Prepare and encode images based on stimulus configuration."""
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
        """Generate responses using the configured VLM provider."""
        self.task_info = loader.get_task_info()
        self.task = loader.get_trials()
        
        for trial in tqdm(self.task, desc="Getting model responses"):
            instruction = trial["prompt"]
            images = self._prepare_images(trial)
            
            conversation = self._format_conversation(instruction, images)
            response_text = self._make_api_call(conversation)
            
            trial["conversation"] = conversation
            trial["model_response"] = response_text
            
        return (self.task_info, self.task)

class OpenAIModelRunner(BaseModelRunner):
    """OpenAI API implementation of the model runner."""
    
    def _initialize_client(self) -> openai.OpenAI:
        """Initialize OpenAI client."""
        return openai.OpenAI(api_key=self.config.api_key)
    
    def _format_conversation(self, instruction: str, images: List[str]) -> List[Dict]:
        """Format conversation for OpenAI API."""
        content = [{"type": "text", "text": instruction}]
        
        # Add contextual text for multi-image scenarios
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
            # For 'one' and 'two' stimulus cases
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
        content = [{"type": "text", "text": instruction}]
        
        for image in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image
                }
            })
        
        return {"role": "user", "content": content}
    
    def _make_api_call(self, conversation: Dict) -> str:
        """Make Anthropic API call."""
        if not hasattr(self, 'client'):
            self.client = self._initialize_client()
            
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[conversation],
            **self.config.additional_params
        )
        
        return response.content[0].text

class GoogleModelRunner(BaseModelRunner):
    """Google Gemini API implementation."""
    
    def _initialize_client(self):
        """Initialize Google client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            return genai.GenerativeModel(self.config.model_name)
        except ImportError:
            raise ImportError("google-generativeai package required for GoogleModelRunner")
    
    def _format_conversation(self, instruction: str, images: List[str]) -> List:
        """Format conversation for Google API."""
        content = [instruction]
        
        for image_b64 in images:
            # Convert base64 to bytes for Google API
            image_bytes = base64.b64decode(image_b64)
            content.append({
                'mime_type': 'image/png',
                'data': image_bytes
            })
        
        return content
    
    def _make_api_call(self, conversation: List) -> str:
        """Make Google API call."""
        if not hasattr(self, 'client'):
            self.client = self._initialize_client()
            
        response = self.client.generate_content(
            conversation,
            generation_config={
                'max_output_tokens': self.config.max_tokens,
                'temperature': self.config.temperature,
                **self.config.additional_params
            }
        )
        
        return response.text

class HuggingFaceModelRunner(BaseModelRunner):
    """Hugging Face Transformers implementation."""
    
    def _initialize_client(self):
        """Initialize Hugging Face client."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            
            processor = AutoProcessor.from_pretrained(self.config.model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            return {"processor": processor, "model": model}
        except ImportError:
            raise ImportError("transformers and torch packages required for HuggingFaceModelRunner")
    
    def _format_conversation(self, instruction: str, images: List[str]) -> Dict:
        """Format conversation for Hugging Face models."""
        from PIL import Image
        import io
        
        # Convert base64 images to PIL Images
        pil_images = []
        for image_b64 in images:
            image_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_images.append(pil_image)
        
        return {
            "text": instruction,
            "images": pil_images
        }
    
    def _make_api_call(self, conversation: Dict) -> str:
        """Make Hugging Face model call."""
        if not hasattr(self, 'client'):
            self.client = self._initialize_client()
        
        processor = self.client["processor"]
        model = self.client["model"]
        
        # Process inputs
        inputs = processor(
            text=conversation["text"],
            images=conversation["images"],
            return_tensors="pt"
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                **self.config.additional_params
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response

# Factory function for easy runner creation
def create_runner(provider: str, config: ModelConfig) -> BaseModelRunner:
    """Factory function to create appropriate runner based on provider."""
    runners = {
        'openai': OpenAIModelRunner,
        'anthropic': AnthropicModelRunner,
        'google': GoogleModelRunner,
        'huggingface': HuggingFaceModelRunner,
    }
    
    if provider not in runners:
        raise ValueError(f"Unsupported provider: {provider}. Available: {list(runners.keys())}")
    
    return runners[provider](config)

# Convenience function for backward compatibility
def create_openai_runner(api_key: str, model_name: str = "gpt-4o") -> OpenAIModelRunner:
    """Create OpenAI runner with simple parameters (backward compatibility)."""
    config = ModelConfig(model_name=model_name, api_key=api_key)
    return OpenAIModelRunner(config)
