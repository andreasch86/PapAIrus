"""Helpers for constructing LLM and embedding providers based on settings."""

try:  # pragma: no cover - optional dependency
    from llama_index.embeddings.gemini import GeminiEmbedding
except ImportError:  # pragma: no cover - optional dependency
    GeminiEmbedding = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:  # pragma: no cover - optional dependency
    OllamaEmbedding = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from llama_index.llms.ollama import Ollama
except ImportError:  # pragma: no cover - optional dependency
    Ollama = None  # type: ignore

from types import SimpleNamespace

import requests

from papairus.settings import ChatCompletionSettings


def _require_dependency(dep, name: str):
    if dep is None:  # pragma: no cover - runtime safeguard
        raise ImportError(
            f"{name} is required for the configured model but is not installed."
        )
    return dep


class VertexGeminiLLM:
    """Minimal Vertex Gemini chat client using an API key."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        timeout: int,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages):
        # Flatten system/user messages into a single prompt string to align with
        # Vertex AI's content payload.
        prompt_parts: list[str] = []
        for message in messages:
            prompt_parts.append(str(message.content))

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "\n\n".join(prompt_parts)}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        endpoint = f"{self.base_url}/publishers/google/models/{self.model}:generateContent"
        response = requests.post(
            endpoint,
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini API")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text_chunks = [part.get("text", "") for part in parts]
        text = "".join(text_chunks)

        message = SimpleNamespace(content=text)
        usage = data.get("usage_metadata", {})
        raw_usage = SimpleNamespace(
            prompt_tokens=usage.get("prompt_token_count", 0),
            completion_tokens=usage.get("candidates_token_count", 0),
            total_tokens=usage.get("total_token_count", 0),
        )
        message.raw = SimpleNamespace(usage=raw_usage)  # type: ignore[attr-defined]
        return message


def build_llm(chat_settings: ChatCompletionSettings):
    """Return a configured LLM implementation for the given settings."""
    if chat_settings.model == "gemini-3-flash":
        api_key = chat_settings.gemini_api_key
        if api_key is None:
            raise ValueError("gemini_api_key must be provided for Gemini models")

        return VertexGeminiLLM(
            api_key=api_key.get_secret_value(),  # type: ignore[union-attr]
            base_url=chat_settings.gemini_base_url,
            model=chat_settings.model,
            temperature=chat_settings.temperature,
            timeout=chat_settings.request_timeout,
        )

    if chat_settings.model == "gemma-local":
        ollama_cls = _require_dependency(Ollama, "llama-index-llms-ollama")
        return ollama_cls(
            model=chat_settings.ollama_model,
            request_timeout=chat_settings.request_timeout,
            temperature=chat_settings.temperature,
            base_url=chat_settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported model configured: {chat_settings.model}")


def build_embedding_model(chat_settings: ChatCompletionSettings):
    """Return an embedding model implementation aligned with the chat settings."""
    if chat_settings.model == "gemini-3-flash":
        gemini_embedding_cls = _require_dependency(
            GeminiEmbedding, "llama-index-embeddings-gemini"
        )
        return gemini_embedding_cls(
            model_name="models/embedding-001",
            api_key=chat_settings.gemini_api_key.get_secret_value(),  # type: ignore[union-attr]
        )

    if chat_settings.model == "gemma-local":
        ollama_embedding_cls = _require_dependency(
            OllamaEmbedding, "llama-index-embeddings-ollama"
        )
        return ollama_embedding_cls(
            model_name=chat_settings.ollama_embedding_model,
            base_url=chat_settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported model configured: {chat_settings.model}")
