"""
This file is a runner for the neuropsych minibench tasks.
As an example, this runner is used to generate responses for the tasks using OpenAI API.
"""

import os
import json
import base64
from typing import List, Dict
from dataclasses import dataclass
import openai
from tqdm import tqdm

class OpenAIModelRunner:
    def __init__(self, api_key, model_name = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name


    @staticmethod
    def encode_image_file(image_path):
        """Encode a local image file to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def generate_response(self, loader):
        """Generate response using OpenAI API."""

        self.task_info = loader.get_task_info()
        self.task = loader.get_trials()

        # If the task has only one stimulus
        if self.task_info["num_stim"] == 'one':

            for trial in tqdm(self.task, desc="Getting model responses"):
                target_image = self.encode_image_file(trial["images"]["target"][0])
                instruction = trial["prompt"]

                conversation = [{"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_image}", "detail": "high"}},

                ]}]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    max_tokens=100
                )
                
                

                trial["conversation"] = conversation
                trial["model_response"] = response.choices[0].message.content

        if self.task_info["num_stim"] == 'two':

            for trial in tqdm(self.task, desc="Getting model responses"):

                image_option1 = self.encode_image_file(trial["images"]["options"][0])
                image_option2 = self.encode_image_file(trial["images"]["options"][1])
                instruction = trial["prompt"]

                conversation = [{"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_option1}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_option2}", "detail": "high"}},

                ]}]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    max_tokens=100
                )
                
                trial["conversation"] = conversation
                trial["model_response"] = response.choices[0].message.content

        if self.task_info["num_stim"] == 'three':

            for trial in tqdm(self.task, desc="Getting model responses"):

                target_image = self.encode_image_file(trial["images"]["target"][0])
                image_option1 = self.encode_image_file(trial["images"]["option_1"][0])
                image_option2 = self.encode_image_file(trial["images"]["option_2"][0])
                instruction = trial["prompt"]

                conversation = [{"role": "user", "content": [

                {"type": "text", "text": instruction},
                {"type": "text", "text": "Here's the target image"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_image}", "detail": "high"}},

                {"type": "text", "text": "Here's the first option"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_option1}", "detail": "high"}},

                {"type": "text", "text": "Here's the second option"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_option2}", "detail": "high"}},

                ]}]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    max_tokens=100
                )
                
                trial["conversation"] = conversation
                trial["model_response"] = response.choices[0].message.content

        if self.task_info["num_stim"] == 'four':    

            for trial in tqdm(self.task, desc="Getting model responses"):
                target_image = self.encode_image_file(trial["images"]["target"][0])
                image_option1 = self.encode_image_file(trial["images"]["option_1"][0])
                image_option2 = self.encode_image_file(trial["images"]["option_2"][0])
                image_option3 = self.encode_image_file(trial["images"]["option_3"][0])
                instruction = trial["prompt"]

                conversation = [{"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "text", "text": "Here's the target image"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_image}", "detail": "high"}},

                {"type": "text", "text": "Here's the first option"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_option1}", "detail": "high"}},

                {"type": "text", "text": "Here's the second option"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_option2}", "detail": "high"}},

                {"type": "text", "text": "Here's the third option"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_option3}", "detail": "high"}}
                ]}]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    max_tokens=100
                )
                
                trial["conversation"] = conversation
                trial["model_response"] = response.choices[0].message.content
        
        return (self.task_info, self.task)