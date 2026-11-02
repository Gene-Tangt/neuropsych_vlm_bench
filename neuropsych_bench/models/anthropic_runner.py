from .base import ModelRunner


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
