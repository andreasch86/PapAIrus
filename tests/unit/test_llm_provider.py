from types import SimpleNamespace

import pytest
import requests

from papairus.llm_provider import VertexGeminiLLM, build_embedding_model, build_llm
from papairus.settings import ChatCompletionSettings


def test_build_llm_uses_local_gemma():
    settings = ChatCompletionSettings(
        model="gemma-local",
        request_timeout=45,
        temperature=0.7,
        gemini_api_key=None,
        ollama_base_url="http://ollama:11434",
        ollama_model="gemma2:2b",
    )

    llm = build_llm(settings)

    from papairus.llm.backends.local_gemma import LocalGemmaBackend

    assert isinstance(llm, LocalGemmaBackend)
    assert llm.model == "gemma2:2b"
    assert llm.base_url == "http://ollama:11434"
    assert llm.temperature == 0.7


def test_build_embedding_model_for_gemma(monkeypatch):
    captured_kwargs = {}

    class FakeOllamaEmbedding:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("papairus.llm_provider.OllamaEmbedding", FakeOllamaEmbedding)

    settings = ChatCompletionSettings(
        model="gemma-local",
        gemini_api_key=None,
        ollama_base_url="http://ollama:11434",
        ollama_embedding_model="gemma-embed:latest",
    )

    embed = build_embedding_model(settings)

    assert isinstance(embed, FakeOllamaEmbedding)
    assert captured_kwargs == {
        "model_name": "gemma-embed:latest",
        "base_url": "http://ollama:11434",
    }


def test_build_embedding_model_for_gemma_missing_dep(monkeypatch):
    monkeypatch.setattr("papairus.llm_provider.OllamaEmbedding", None)

    settings = ChatCompletionSettings(model="gemma-local")

    with pytest.raises(ImportError) as excinfo:
        build_embedding_model(settings)

    assert "pip install \"llama-index-embeddings-ollama>=0.3.0\"" in str(excinfo.value)


def test_build_llm_uses_gemini(monkeypatch):
    settings = ChatCompletionSettings(
        model="gemini-2.5-flash",
        gemini_api_key="dummy-key",
        request_timeout=30,
        temperature=0.2,
    )

    llm = build_llm(settings)

    assert llm.model == "gemini-2.5-flash"
    assert llm.api_key == "dummy-key"
    assert llm.timeout == 30


def test_build_llm_rejects_missing_gemini_api_key():
    settings = ChatCompletionSettings.model_construct(
        model="gemini-3-flash", gemini_api_key=None
    )

    with pytest.raises(ValueError, match="gemini_api_key must be provided"):
        build_llm(settings)


def test_build_llm_rejects_unknown_model():
    settings = ChatCompletionSettings.model_construct(
        model="unknown-model", gemini_api_key=None
    )

    with pytest.raises(ValueError, match="Unsupported model"):
        build_llm(settings)


def test_build_embedding_model_for_gemini(monkeypatch):
    captured_kwargs = {}

    class FakeGeminiEmbedding:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("papairus.llm_provider.GeminiEmbedding", FakeGeminiEmbedding)

    settings = ChatCompletionSettings(model="gemini-1.5-flash", gemini_api_key="sekret")

    embed = build_embedding_model(settings)

    assert isinstance(embed, FakeGeminiEmbedding)
    assert captured_kwargs == {
        "model_name": "models/embedding-001",
        "api_key": "sekret",
    }


def test_build_embedding_model_rejects_unknown_model():
    settings = ChatCompletionSettings.model_construct(
        model="unknown-model", gemini_api_key=None
    )

    with pytest.raises(ValueError, match="Unsupported model"):
        build_embedding_model(settings)


def test_vertex_gemini_llm_returns_text_and_usage(monkeypatch):
    def fake_json():
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hello"}, {"text": " world"}]},
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 1,
                "candidates_token_count": 2,
                "total_token_count": 3,
            },
        }

    class FakeResponse:
        status_code = 200

        def __init__(self):
            self.url = "https://example.com/generate"

        def raise_for_status(self):
            return None

        def json(self):
            return fake_json()

    def fake_post(*_args, **_kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    llm = VertexGeminiLLM(
        api_key="dummy",
        base_url="https://aiplatform.googleapis.com/v1",
        model="gemini-2.5-flash",
        temperature=0.1,
        timeout=5,
    )

    result = llm.chat([SimpleNamespace(content="hi")])

    assert result.message.content == "hello world"
    assert result.raw.usage.prompt_tokens == 1
    assert result.raw.usage.completion_tokens == 2
    assert result.raw.usage.total_tokens == 3


def test_vertex_gemini_llm_raises_clear_error_on_404(monkeypatch):
    class FakeResponse:
        def __init__(self):
            self.status_code = 404
            self.text = "model not found"
            self.url = "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash:generateContent"

        def raise_for_status(self):
            raise requests.HTTPError(response=self)

        def json(self):  # pragma: no cover - not used when raising
            return {}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    llm = VertexGeminiLLM(
        api_key="dummy",
        base_url="https://aiplatform.googleapis.com/v1",
        model="gemini-3-flash",
        temperature=0.0,
        timeout=30,
    )

    with pytest.raises(ValueError) as excinfo:
        llm.chat([SimpleNamespace(content="hello")])

    assert "Gemini model not found" in str(excinfo.value)


def test_vertex_gemini_llm_raises_when_no_candidates(monkeypatch):
    class FakeResponse:
        status_code = 200
        url = "https://example.com/generate"

        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": []}

    monkeypatch.setattr("requests.post", lambda *_args, **_kwargs: FakeResponse())

    llm = VertexGeminiLLM(
        api_key="dummy",
        base_url="https://aiplatform.googleapis.com/v1",
        model="gemini-2.5-flash",
        temperature=0.1,
        timeout=5,
    )

    with pytest.raises(RuntimeError, match="No candidates returned"):
        llm.chat([SimpleNamespace(content="hi")])
