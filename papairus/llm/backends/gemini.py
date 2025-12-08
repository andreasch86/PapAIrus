"""Gemini backend implementation for the shared LLM interface."""

from __future__ import annotations

from typing import Any, Sequence

import requests
from requests import HTTPError

from papairus.llm.backends.base import ChatMessage, LLMBackend, LLMMetadata, LLMResponse, LLMUsage


class GeminiBackend(LLMBackend):
    def __init__(
        self,
        *,
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
        self._metadata = LLMMetadata(
            model_name=model, context_window=8192, num_output=1024, type="gemini"
        )

    def generate_docstring(
        self,
        code_snippet: str,
        *,
        style: str = "google",
        existing_docstring: str | None = None,
    ) -> str:
        header = (
            "Generate a concise Google-style Python docstring."
            " Include Args/Returns/Raises as appropriate and avoid conversational preambles."
        )
        if existing_docstring:
            header += " Update the existing docstring to fill missing sections."
        prompt = "\n\n".join([header, "Code:", code_snippet])
        response = self.generate_response([ChatMessage(role="user", content=prompt)])
        return response.message.content

    def generate_response(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        prompt_parts: list[str] = [message.content for message in messages]
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
        try:
            response.raise_for_status()
        except HTTPError as exc:  # pragma: no cover - defensive runtime path
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 404:
                raise ValueError(
                    "Gemini model not found. Confirm the model name (e.g., "
                    "gemini-2.0-flash, gemini-2.5-flash, gemini-1.5-pro-latest) "
                    "and that it is available for API-key access. "
                    f"Endpoint: {endpoint}"
                ) from exc
            raise

        data: dict[str, Any] = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini API")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text_chunks = [part.get("text", "") for part in parts]
        text = "".join(text_chunks)

        usage_data = data.get("usage_metadata", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_token_count", 0),
            completion_tokens=usage_data.get("candidates_token_count", 0),
            total_tokens=usage_data.get("total_token_count", 0),
        )
        return LLMResponse(message=ChatMessage(role="assistant", content=text), usage=usage)
