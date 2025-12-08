"""Helpers for constructing LLM and embedding providers based on settings."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from llama_index.embeddings.gemini import GeminiEmbedding
except ImportError:  # pragma: no cover - optional dependency
    GeminiEmbedding = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:  # pragma: no cover - optional dependency
    OllamaEmbedding = None  # type: ignore

from papairus.llm.backends.base import LLMBackend
from papairus.llm.backends.codegemma import CodegemmaBackend
from papairus.llm.backends.gemini import GeminiBackend
from papairus.settings import ChatCompletionSettings


def _require_dependency(dep, name: str):
    if dep is None:  # pragma: no cover - runtime safeguard
        raise ImportError(f"{name} is required for the configured model but is not installed.")
    return dep


def _resolve_engine(settings: ChatCompletionSettings) -> str:
    if getattr(settings, "engine", None):
        return settings.engine
    if settings.model.startswith("gemini-"):
        return "gemini"
    if settings.model == "codegemma":
        return "codegemma"
    raise ValueError(f"Unsupported model configured: {settings.model}")


def build_llm(settings: ChatCompletionSettings) -> LLMBackend:
    """Return a configured LLM backend for the given settings."""

    engine = _resolve_engine(settings)
    if engine == "gemini":
        api_key = settings.gemini_api_key
        if api_key is None:
            raise ValueError("gemini_api_key must be provided for Gemini models")

        return GeminiBackend(
            api_key=api_key.get_secret_value(),  # type: ignore[union-attr]
            base_url=settings.gemini_base_url,
            model=settings.model,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
        )

    if engine == "codegemma":
        return CodegemmaBackend(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            request_timeout=settings.request_timeout,
            auto_pull=settings.ollama_auto_pull,
        )

    raise ValueError(f"Unsupported engine configured: {engine}")


def build_embedding_model(chat_settings: ChatCompletionSettings):
    """Return an embedding model implementation aligned with the chat settings."""
    engine = _resolve_engine(chat_settings)
    if engine == "gemini":
        gemini_embedding_cls = _require_dependency(GeminiEmbedding, "llama-index-embeddings-gemini")
        return gemini_embedding_cls(
            model_name="models/embedding-001",
            api_key=chat_settings.gemini_api_key.get_secret_value(),  # type: ignore[union-attr]
        )

    if engine == "codegemma":
        try:
            ollama_embedding_cls = _require_dependency(
                OllamaEmbedding, "llama-index-embeddings-ollama"
            )
        except ImportError as exc:  # pragma: no cover - defensive runtime message
            raise ImportError(
                "llama-index-embeddings-ollama is required for chat-with-repo. "
                'Install it with `pip install "llama-index-embeddings-ollama>=0.3.0"`.'
            ) from exc

        return ollama_embedding_cls(
            model_name=chat_settings.ollama_embedding_model,
            base_url=chat_settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported model configured: {chat_settings.model}")
