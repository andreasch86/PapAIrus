import types

import pytest
import requests

from papairus.llm.backends.base import ChatMessage
from papairus.llm.backends.local_gemma import LocalGemmaBackend
from papairus.llm_provider import build_llm
from papairus.settings import ChatCompletionSettings


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)  # type: ignore[name-defined]

    def json(self):
        return self._data


def test_build_llm_returns_gemini_backend():
    settings = ChatCompletionSettings(
        model="gemini-2.5-flash", gemini_api_key="secret", temperature=0.1
    )
    backend = build_llm(settings)
    from papairus.llm.backends.gemini import GeminiBackend

    assert isinstance(backend, GeminiBackend)


@pytest.mark.parametrize("auto_pull", [True, False])
def test_local_gemma_autopull(monkeypatch, auto_pull):
    pull_called = {"pull": 0, "chat": 0}
    tag_states = iter([
        {"models": []},
        {"models": [{"name": "gemma:2b"}]},
    ])

    def fake_get(url, timeout):
        return FakeResponse(next(tag_states, {"models": [{"name": "gemma:2b"}]}))

    def fake_post(url, json, timeout):
        if url.endswith("/api/pull"):
            pull_called["pull"] += 1
            return FakeResponse({}, 200)
        pull_called["chat"] += 1
        return FakeResponse(
            {"message": {"content": "ok"}, "eval_count": 2, "prompt_eval_count": 1}, 200
        )

    monkeypatch.setattr("papairus.llm.backends.local_gemma.requests.get", fake_get)
    monkeypatch.setattr("papairus.llm.backends.local_gemma.requests.post", fake_post)

    backend = LocalGemmaBackend(
        model="gemma:2b", base_url="http://localhost:11434", auto_pull=auto_pull
    )

    response = backend.generate_response([ChatMessage(role="user", content="hi")])
    assert response.message.content == "ok"
    if auto_pull:
        assert pull_called["pull"] == 1
    else:
        assert pull_called["pull"] == 0
    assert pull_called["chat"] == 1
