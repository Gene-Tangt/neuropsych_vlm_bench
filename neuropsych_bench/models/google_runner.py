import base64

from .base import ModelRunner


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
