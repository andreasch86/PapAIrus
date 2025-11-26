import pytest

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


def test_build_llm_uses_gemini(monkeypatch):
    captured_kwargs = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("papairus.llm_provider.Gemini", FakeGemini)

    settings = ChatCompletionSettings(
        model="gemini-3.5-flash",
        gemini_api_key="dummy-key",
        request_timeout=30,
        temperature=0.2,
    )

    llm = build_llm(settings)

    assert isinstance(llm, FakeGemini)
    assert captured_kwargs["api_key"] == "dummy-key"
    assert captured_kwargs["model"] == "gemini-3.5-flash"
    assert captured_kwargs["timeout"] == 30
