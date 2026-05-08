from .base import LLMClient, LLMResponse
from .genai_config import GenAIConfigManager
import time

from openai import OpenAI


class LocalServerClient(LLMClient):
    """Client for a locally hosted OpenAI-compatible LLM server (e.g. llama.cpp, LM Studio, Ollama).

    :param base_url: Base URL of the server's OpenAI-compatible API
    :param api_key: API key (most local servers accept any non-empty string)
    :param model: Model identifier to pass in requests
    :param config_manager: GenAIConfigManager instance (optional)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "dummy_key",
        model: str = "model-identifier",
        config_manager: GenAIConfigManager = None,
    ):
        if config_manager is None:
            config_manager = GenAIConfigManager()
        self.config_manager = config_manager
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, messages, temperature: float = 0.7, **kwargs) -> LLMResponse:
        """Generate text using the local server.

        :param messages: List of message dicts with 'role' and 'content' keys,
            or a plain string (wrapped into a user message automatically)
        :param temperature: Sampling temperature
        :return: LLMResponse with generated text, token count, and latency
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        t0 = time.time()
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return LLMResponse(
            text=completion.choices[0].message.content,
            tokens=completion.usage.total_tokens if completion.usage else -1,
            latency=time.time() - t0,
            raw=completion.model_dump(),
        )
