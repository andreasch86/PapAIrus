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
    from llama_index.llms.gemini import Gemini
except ImportError:  # pragma: no cover - optional dependency
    Gemini = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from llama_index.llms.ollama import Ollama
except ImportError:  # pragma: no cover - optional dependency
    Ollama = None  # type: ignore

from papairus.settings import ChatCompletionSettings


def _require_dependency(dep, name: str):
    if dep is None:  # pragma: no cover - runtime safeguard
        raise ImportError(
            f"{name} is required for the configured model but is not installed."
        )
    return dep


def build_llm(chat_settings: ChatCompletionSettings):
    """Return a configured LLM implementation for the given settings."""
    if chat_settings.model == "gemini-3-flash":
        gemini_cls = _require_dependency(Gemini, "llama-index-llms-gemini")
        return Gemini(
            api_key=chat_settings.gemini_api_key.get_secret_value(),  # type: ignore[union-attr]
            timeout=chat_settings.request_timeout,
            model=chat_settings.model,
            temperature=chat_settings.temperature,
            max_output_tokens=None,
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
