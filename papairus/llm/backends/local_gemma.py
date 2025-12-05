"""Local Gemma backend leveraging the Ollama HTTP API with auto-pull support."""

from __future__ import annotations

import time
from typing import Any, Sequence

import requests
from requests import HTTPError

from papairus.llm.backends.base import ChatMessage, LLMBackend, LLMResponse, LLMUsage


class LocalGemmaBackend(LLMBackend):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        request_timeout: int = 60,
        auto_pull: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = request_timeout
        self.auto_pull = auto_pull

    def generate_docstring(
        self,
        code_snippet: str,
        *,
        style: str = "google",
        existing_docstring: str | None = None,
    ) -> str:
        header = (
            "You are a code documentation assistant."
            " Produce a Python docstring in Google style with Args/Returns/Raises."
            " Do not add prose outside the docstring body."
        )
        if existing_docstring:
            header += (
                " Update the existing docstring to cover missing sections without changing intent."
            )
        prompt = "\n\n".join([header, "Code:", code_snippet])
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.generate_response(messages)
        return response.message.content

    def generate_response(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        self._ensure_model()
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "options": {"temperature": self.temperature},
            "stream": False,
        }
        endpoint = f"{self.base_url}/api/chat"
        response = requests.post(endpoint, json=payload, timeout=self.timeout)
        try:
            response.raise_for_status()
        except HTTPError as exc:  # pragma: no cover - defensive runtime path
            raise RuntimeError(f"Ollama chat request failed: {exc}") from exc

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:  # pragma: no cover - defensive runtime path
            truncated = response.text[:200]
            raise RuntimeError(
                "Ollama chat returned non-JSON response; ensure streaming is disabled "
                f"and the model is available. Response snippet: {truncated}"
            ) from exc
        message = data.get("message", {})
        content = message.get("content", "")
        usage_data = data.get("eval_count", 0)
        prompt_count = data.get("prompt_eval_count", 0)
        usage = LLMUsage(
            prompt_tokens=prompt_count,
            completion_tokens=usage_data,
            total_tokens=prompt_count + usage_data,
        )
        return LLMResponse(message=ChatMessage(role="assistant", content=content), usage=usage)

    def _ensure_model(self) -> None:
        if not self.auto_pull:
            return
        if self._model_available():
            return
        pull_endpoint = f"{self.base_url}/api/pull"
        response = requests.post(pull_endpoint, json={"name": self.model}, timeout=self.timeout)
        try:
            response.raise_for_status()
        except HTTPError as exc:  # pragma: no cover - defensive runtime path
            raise RuntimeError(f"Failed to pull Ollama model {self.model}: {exc}") from exc
        # Wait briefly for the model to register
        self._wait_for_model()

    def _model_available(self) -> bool:
        tags_endpoint = f"{self.base_url}/api/tags"
        response = requests.get(tags_endpoint, timeout=self.timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        models = data.get("models", [])
        return any(entry.get("name") == self.model for entry in models)

    def _wait_for_model(self) -> None:
        # Poll a couple of times to account for async pull completion
        for _ in range(3):
            if self._model_available():
                return
            time.sleep(1)
