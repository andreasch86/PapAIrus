import pytest
import requests
from types import SimpleNamespace

from papairus.llm_provider import build_embedding_model, build_llm
from papairus.settings import ChatCompletionSettings


def test_build_llm_uses_ollama(monkeypatch):
    captured_kwargs = {}

    class FakeOllama:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("papairus.llm_provider.Ollama", FakeOllama)

    settings = ChatCompletionSettings(
        model="gemma-local",
        request_timeout=45,
        temperature=0.7,
        gemini_api_key=None,
        ollama_base_url="http://ollama:11434",
        ollama_model="gemma2:2b",
    )

    llm = build_llm(settings)

    assert isinstance(llm, FakeOllama)
    assert captured_kwargs == {
        "model": "gemma2:2b",
        "request_timeout": 45,
        "temperature": 0.7,
        "base_url": "http://ollama:11434",
    }


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


def test_vertex_gemini_llm_raises_clear_error_on_404(monkeypatch):
    from papairus.llm_provider import VertexGeminiLLM

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
