from .base import ModelRunner


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
