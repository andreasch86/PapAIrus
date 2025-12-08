import types

import pytest
import requests

from papairus.llm.backends.base import ChatMessage
from papairus.llm.backends.local_gemma import LocalGemmaBackend
from papairus.llm_provider import build_llm
from papairus.settings import ChatCompletionSettings


class FakeResponse:
    def __init__(self, data, status_code=200, *, json_raises: bool = False):
        self._data = data
        self.status_code = status_code
        self._json_raises = json_raises
        self.text = data if isinstance(data, str) else str(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)  # type: ignore[name-defined]

    def json(self):
        if self._json_raises:
            raise ValueError("not json")
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
        {"models": [{"name": "codegemma:instruct"}]},
    ])

    def fake_get(url, timeout):
        return FakeResponse(next(tag_states, {"models": [{"name": "codegemma:instruct"}]}))

    def fake_post(url, json, timeout):
        if url.endswith("/api/pull"):
            pull_called["pull"] += 1
            return FakeResponse({}, 200)
        pull_called["chat"] += 1
        assert json.get("stream") is False
        return FakeResponse(
            {"message": {"content": "ok"}, "eval_count": 2, "prompt_eval_count": 1}, 200
        )

    monkeypatch.setattr("papairus.llm.backends.local_gemma.requests.get", fake_get)
    monkeypatch.setattr("papairus.llm.backends.local_gemma.requests.post", fake_post)

    backend = LocalGemmaBackend(
        model="codegemma:instruct", base_url="http://localhost:11434", auto_pull=auto_pull
    )

    response = backend.generate_response([ChatMessage(role="user", content="hi")])
    assert response.message.content == "ok"
    if auto_pull:
        assert pull_called["pull"] == 1
    else:
        assert pull_called["pull"] == 0
    assert pull_called["chat"] == 1


def test_local_gemma_non_json_response(monkeypatch):
    def fake_get(url, timeout):
        return FakeResponse({"models": [{"name": "codegemma:instruct"}]})

    def fake_post(url, json, timeout):
        return FakeResponse("not json\n{ malformed }", json_raises=True)

    monkeypatch.setattr("papairus.llm.backends.local_gemma.requests.get", fake_get)
    monkeypatch.setattr("papairus.llm.backends.local_gemma.requests.post", fake_post)

    backend = LocalGemmaBackend(model="codegemma:instruct")

    with pytest.raises(RuntimeError):
        backend.generate_response([ChatMessage(role="user", content="hi")])


def test_local_gemma_normalizes_non_string_content(monkeypatch):
    def fake_get(url, timeout):
        return FakeResponse({"models": [{"name": "codegemma:instruct"}]})

    captured_payload = {}

    def fake_post(url, json, timeout):
        captured_payload["messages"] = json["messages"]
        return FakeResponse(
            {"message": {"content": "ok"}, "eval_count": 2, "prompt_eval_count": 1}, 200
        )

    monkeypatch.setattr("papairus.llm.backends.local_gemma.requests.get", fake_get)
    monkeypatch.setattr("papairus.llm.backends.local_gemma.requests.post", fake_post)

    class DummyTemplate:
        def __str__(self):  # pragma: no cover - trivial
            return "templated"

    backend = LocalGemmaBackend(model="codegemma:instruct")
    response = backend.generate_response(
        [ChatMessage(role="user", content=DummyTemplate())]
    )

    assert response.message.content == "ok"
    assert captured_payload["messages"] == [{"role": "user", "content": "templated"}]


def test_llm_metadata_defaults():
    backend = LocalGemmaBackend(model="codegemma:instruct")
    metadata = backend.metadata

    assert metadata.context_window == 8192
    assert metadata.num_output == 1024
    assert metadata.model_name == "codegemma:instruct"
    assert metadata.is_chat_model is True
