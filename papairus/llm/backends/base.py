"""Base interfaces and lightweight message/response types for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterable, Sequence


@dataclass
class ChatMessage:
    """Minimal chat message container for backend-agnostic prompts."""

    role: str
    content: str


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMMetadata:
    """Lightweight metadata container matching llama-index expectations."""

    model_name: str
    context_window: int = 8192
    num_output: int = 1024
    type: str = "custom"
    is_chat_model: bool = True


@dataclass
class LLMResponse:
    """Normalized response with llama-index compatible accessors."""

    message: ChatMessage
    usage: LLMUsage = field(default_factory=LLMUsage)

    @property
    def raw(self) -> SimpleNamespace:  # pragma: no cover - simple passthrough
        return SimpleNamespace(usage=self.usage)

    @property
    def text(self) -> str:
        return self.message.content


class LLMBackend(ABC):
    """Abstract base for all LLM backends."""

    def __init__(self) -> None:
        self._system_prompt: str = ""

    @property
    def metadata(self) -> LLMMetadata:
        """Return llama-index compatible metadata for the backend."""

        if hasattr(self, "_metadata"):
            meta: LLMMetadata = getattr(self, "_metadata")  # type: ignore[attr-defined]
            if meta.context_window is None:
                meta.context_window = LLMMetadata.context_window
            if meta.num_output is None:
                meta.num_output = LLMMetadata.num_output
            if meta.is_chat_model is None:
                meta.is_chat_model = True
            return meta

        model_name = getattr(self, "model", "unknown")
        return LLMMetadata(model_name=model_name, type=self.__class__.__name__.lower())

    @property
    def system_prompt(self) -> str:
        """System prompt hint for llama-index prompt helpers."""

        return getattr(self, "_system_prompt", "")

    def update_system_prompt(self, prompt: str) -> None:
        """Persist a system prompt for downstream prompt helpers."""

        self._system_prompt = prompt

    @abstractmethod
    def generate_response(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """Return a normalized chat response for the provided messages."""

    @abstractmethod
    def generate_docstring(
        self,
        code_snippet: str,
        *,
        style: str = "google",
        existing_docstring: str | None = None,
    ) -> str:
        """Generate a docstring for the given code snippet in the requested style."""

    def chat(self, messages: Sequence[ChatMessage], **_: object):
        """Compatibility shim mirroring llama-index ChatResponse shape."""

        normalized = list(self._normalize_messages(messages))
        return self.generate_response(normalized)

    def complete(self, prompt: str, **_: object):
        return SimpleNamespace(text=self.predict(prompt))

    def predict(self, prompt: str, **_: object) -> str:
        """Compatibility shim for llama-index completion calls."""

        messages: list[ChatMessage] = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        response = self.generate_response(messages)
        return response.message.content

    def _normalize_messages(
        self, messages: Sequence[ChatMessage | SimpleNamespace | dict | str]
    ) -> Iterable[ChatMessage]:
        for message in messages:
            if isinstance(message, ChatMessage):
                yield message
                continue
            if isinstance(message, str):
                yield ChatMessage(role="user", content=message)
                continue
            content = getattr(message, "content", None)
            role = getattr(message, "role", None)
            if content is None:
                content = message.get("content") if isinstance(message, dict) else None
            if role is None:
                role = message.get("role") if isinstance(message, dict) else "user"
            if content is None or role is None:
                raise ValueError("Each message must include role and content")
            yield ChatMessage(role=str(role), content=str(content))
