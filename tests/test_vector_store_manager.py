import asyncio
import types

import pytest

pytest.importorskip("chromadb")

from papairus.chat_with_repo.vector_store_manager import VectorStoreManager
from papairus.exceptions import EmbeddingServiceError, MissingEmbeddingModelError


class DummyDocument:
    def __init__(self, text, extra_info=None):
        self.text = text
        self.extra_info = extra_info or {}


class DummyDocumentNoText:
    def __init__(self, text=None, extra_info=None):
        self.extra_info = extra_info or {}


class DummySplitter:
    def __init__(self, *args, **kwargs):
        self.chunk_size = kwargs.get("chunk_size")

    def get_nodes_from_documents(self, docs):
        return [f"node-{idx}" for idx, _ in enumerate(docs)]


class DummyPersistentClient:
    def __init__(self, path):
        self.path = path

    def get_or_create_collection(self, name):
        return {"name": name}


class DummyChromaVectorStore:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection


class DummyStorageContext:
    @classmethod
    def from_defaults(cls, vector_store):
        ctx = cls()
        ctx.vector_store = vector_store
        return ctx


class DummyVectorStoreIndex:
    def __init__(self, nodes, storage_context=None, embed_model=None):
        self.nodes = nodes
        self.storage_context = storage_context
        self.embed_model = embed_model


class DummyRetriever:
    def __init__(self, index=None, similarity_top_k=None, embed_model=None):
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.embed_model = embed_model


class DummyResponseSynthesizer:
    def __init__(self, llm=None):
        self.llm = llm


class DummyQueryEngine:
    def __init__(self, retriever=None, response_synthesizer=None):
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer

    def query(self, query):
        return types.SimpleNamespace(response="ok", metadata={})


class DummyEmbedModel:
    def __init__(self, *, model_name="dummy-embed", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def get_text_embedding(self, _text):
        return [0.1, 0.2]


class BatchOnlyEmbedModel:
    def __init__(self):
        self.calls = 0

    def get_text_embedding_batch(self, texts):
        self.calls += 1
        return [[0.1] * len(texts)]


class ChunkingEmbedModel:
    def __init__(self, limit=2):
        self.limit = limit
        self.calls = []

    def get_text_embedding_batch(self, texts):
        if len(texts) > self.limit:
            raise RuntimeError("batch too large")
        self.calls.append(list(texts))
        return [[0.5] for _ in texts]


class FakeResponseError(Exception):
    def __init__(self, message="model missing", status_code=404):
        super().__init__(message)
        self.status_code = status_code


class ExplodingSplitter:
    def __init__(self, *args, **kwargs):
        self.chunk_size = kwargs.get("chunk_size")

    def get_nodes_from_documents(self, _docs):
        raise ValueError("boom")


@pytest.fixture
def patched_manager(monkeypatch):
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.Document", DummyDocument
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SemanticSplitterNodeParser",
        DummySplitter,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SentenceSplitter",
        DummySplitter,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.chromadb.PersistentClient",
        DummyPersistentClient,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.ChromaVectorStore",
        DummyChromaVectorStore,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.StorageContext",
        DummyStorageContext,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorStoreIndex",
        DummyVectorStoreIndex,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorIndexRetriever",
        DummyRetriever,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.get_response_synthesizer",
        lambda llm=None: DummyResponseSynthesizer(llm),
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.RetrieverQueryEngine",
        DummyQueryEngine,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager._list_ollama_models",
        lambda _base_url: ["dummy-embed"],
    )
    manager = VectorStoreManager(top_k=1, llm="llm", embed_model=DummyEmbedModel())
    return manager


def test_create_vector_store_uses_text_attribute(patched_manager):
    md_contents = ["hello world"]
    meta_data = [{"source": "test"}]

    patched_manager.create_vector_store(md_contents, meta_data)

    assert patched_manager.query_engine is not None
    assert patched_manager.query_engine.retriever.index.nodes == ["node-0"]


def test_create_vector_store_handles_missing_text(monkeypatch, patched_manager):
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.Document",
        DummyDocumentNoText,
    )
    patched_manager.create_vector_store(["content"], [{"source": "test"}])

    assert patched_manager.query_engine is not None


def test_fallback_splitter_scales_with_metadata(monkeypatch, patched_manager):
    long_meta = "x" * 5000
    captured = {}

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SemanticSplitterNodeParser",
        ExplodingSplitter,
    )

    def capturing_splitter(**kwargs):
        captured["chunk_size"] = kwargs["chunk_size"]
        return DummySplitter(**kwargs)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SentenceSplitter",
        lambda **kwargs: capturing_splitter(**kwargs),
    )

    patched_manager.create_vector_store(["content"], [long_meta])

    assert captured["chunk_size"] >= len(long_meta)


def test_query_store_without_engine_returns_empty():
    manager = VectorStoreManager(top_k=1, llm="llm", embed_model="embed")

    assert manager.query_store("anything") == []


def test_extract_doc_text_handles_missing_and_callable():
    from papairus.chat_with_repo.vector_store_manager import _extract_doc_text

    class WithText:
        text = "hello"

    class WithGetter:
        def __init__(self, value=None, should_raise=False):
            self.value = value
            self.should_raise = should_raise

        def get_text(self):
            if self.should_raise:
                raise ValueError("boom")
            return self.value

    assert _extract_doc_text(WithText()) == "hello"
    assert _extract_doc_text(WithGetter("alt")) == "alt"
    assert _extract_doc_text(WithGetter()) == ""
    assert _extract_doc_text(WithGetter(should_raise=True)) == ""


def test_create_vector_store_skips_when_missing_data(patched_manager):
    patched_manager.create_vector_store([], [])

    assert patched_manager.query_engine is None


def test_create_vector_store_handles_empty_nodes(monkeypatch, patched_manager):
    class EmptySplitter:
        def __init__(self, *args, **kwargs):
            self.chunk_size = kwargs.get("chunk_size")

        def get_nodes_from_documents(self, docs):
            return []

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SemanticSplitterNodeParser",
        EmptySplitter,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SentenceSplitter",
        EmptySplitter,
    )

    patched_manager.create_vector_store(["content"], [{"source": "test"}])

    assert patched_manager.query_engine is None


def test_query_store_with_engine(monkeypatch, patched_manager):
    patched_manager.query_engine = DummyQueryEngine()

    assert patched_manager.query_store("query") == [{"text": "ok", "metadata": {}}]


def test_missing_embedding_model_raises_clear_error(monkeypatch, patched_manager):
    patched_manager.embed_model = DummyEmbedModel(model_name="missing-embed")

    def failing_index(*_args, **_kwargs):
        raise FakeResponseError("model \"missing-embed\" not found", status_code=404)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorStoreIndex", failing_index
    )

    with pytest.raises(MissingEmbeddingModelError) as excinfo:
        patched_manager.create_vector_store(["content"], [{"source": "test"}])

    assert "missing-embed" in str(excinfo.value)


def test_non_embedding_error_is_propagated(monkeypatch, patched_manager):
    def failing_index(*_args, **_kwargs):
        raise ValueError("explode")

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorStoreIndex", failing_index
    )

    with pytest.raises(ValueError):
        patched_manager.create_vector_store(["content"], [{"source": "test"}])


def test_missing_embedding_model_without_base_url(monkeypatch, patched_manager):
    patched_manager.embed_model = DummyEmbedModel(model_name="missing-embed", base_url=None)

    def failing_index(*_args, **_kwargs):
        raise FakeResponseError("model \"missing-embed\" not found", status_code=404)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorStoreIndex", failing_index
    )

    with pytest.raises(MissingEmbeddingModelError) as excinfo:
        patched_manager.create_vector_store(["content"], [{"source": "test"}])

    assert "pull missing-embed" in str(excinfo.value)


def test_missing_embedding_model_error_bubbles_from_index(monkeypatch, patched_manager):
    patched_manager.embed_model = DummyEmbedModel(model_name="missing-embed")

    def failing_index(*_args, **_kwargs):
        raise FakeResponseError("model \"missing-embed\" not found", status_code=404)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager._list_ollama_models",
        lambda _base_url: ["missing-embed"],
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorStoreIndex", failing_index
    )

    with pytest.raises(MissingEmbeddingModelError):
        patched_manager.create_vector_store(["content"], [{"source": "test"}])


def test_preflight_embedding_check_runs_once(monkeypatch, patched_manager):
    call_count = {"count": 0}

    class FailingEmbedModel(DummyEmbedModel):
        def get_text_embedding(self, _text):
            call_count["count"] += 1
            raise FakeResponseError(
                "model \"missing-embed\" not found", status_code=404
            )

    patched_manager.embed_model = FailingEmbedModel(model_name="missing-embed")

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager._list_ollama_models",
        lambda _base_url: ["missing-embed"],
    )

    with pytest.raises(MissingEmbeddingModelError):
        patched_manager.create_vector_store(["content"], [{"source": "test"}])

    # Should short-circuit before any splitter/index calls repeat embed attempts.
    assert call_count["count"] == 1


def test_create_vector_store_retries_with_smaller_batches(monkeypatch, patched_manager):
    attempts = {"count": 0}

    def conditional_index(nodes, storage_context=None, embed_model=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise FakeResponseError(
                "do embedding request: Post \"http://localhost:11434/embedding\": EOF",
                status_code=500,
            )
        return DummyVectorStoreIndex(nodes, storage_context, embed_model)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorStoreIndex", conditional_index
    )

    patched_manager.create_vector_store(["content"], [{"source": "test"}])

    assert attempts["count"] == 2
    assert getattr(patched_manager.embed_model, "embed_batch_size", None) == 1
    assert patched_manager.query_engine is not None


def test_embedding_error_propagates_when_already_min_batch(monkeypatch, patched_manager):
    from papairus.chat_with_repo.vector_store_manager import _wrap_chunking_embed_model

    patched_manager.embed_model = _wrap_chunking_embed_model(
        DummyEmbedModel(), max_batch_size=1
    )

    def failing_index(*_args, **_kwargs):
        raise FakeResponseError(
            "do embedding request: Post \"http://localhost:11434/embedding\": EOF",
            status_code=500,
        )

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorStoreIndex", failing_index
    )

    with pytest.raises(EmbeddingServiceError):
        patched_manager.create_vector_store(["content"], [{"source": "test"}])


def test_chunking_wrapper_splits_batches():
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    embed_model = ChunkingEmbedModel(limit=2)
    wrapped = _wrap_chunking_embed_model(embed_model, max_batch_size=2)

    result = wrapped.get_text_embedding_batch(["a", "b", "c", "d", "e"])

    assert len(result) == 5
    assert embed_model.calls == [["a", "b"], ["c", "d"], ["e"]]


def test_chunking_wrapper_noop_when_already_wrapped():
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    embed_model = ChunkingEmbedModel(limit=3)
    wrapped = _wrap_chunking_embed_model(embed_model, max_batch_size=2)

    assert _wrap_chunking_embed_model(wrapped) is wrapped


def test_chunking_wrapper_respects_min_batch_size():
    from papairus.chat_with_repo.vector_store_manager import (
        _ChunkingEmbeddingWrapper,
        _wrap_chunking_embed_model,
    )

    with pytest.raises(ValueError):
        _wrap_chunking_embed_model(ChunkingEmbedModel(), max_batch_size=0)

    with pytest.raises(ValueError):
        _ChunkingEmbeddingWrapper(ChunkingEmbedModel(), max_batch_size=0)


def test_chunking_wrapper_falls_back_to_single_embed():
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    class SingleEmbed:
        def __init__(self):
            self.calls = []

        def get_text_embedding(self, text):
            self.calls.append(text)
            return [0.7]

    wrapped = _wrap_chunking_embed_model(SingleEmbed(), max_batch_size=3)

    result = wrapped.get_text_embedding_batch(["x", "y"])

    assert result == [[0.7], [0.7]]


def test_chunking_wrapper_raises_when_missing_embedding_apis():
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    class NoEmbed:
        pass

    wrapped = _wrap_chunking_embed_model(NoEmbed())

    with pytest.raises(AttributeError):
        wrapped.get_text_embedding_batch(["x"])


def test_chunking_wrapper_get_text_embedding_missing_method():
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    class BatchOnly:
        def get_text_embedding_batch(self, texts):
            return [[1.0] for _ in texts]

    wrapped = _wrap_chunking_embed_model(BatchOnly())

    with pytest.raises(AttributeError):
        wrapped.get_text_embedding("nope")


def test_chunking_wrapper_delegates_attributes():
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    class EmbedWithAttrs:
        def __init__(self):
            self.custom = "ok"

        def get_text_embedding(self, text):
            return [0.2]

    wrapped = _wrap_chunking_embed_model(EmbedWithAttrs())

    assert wrapped.custom == "ok"
    # Accessing a private attribute should bypass delegation.
    assert wrapped.__getattr__("_wrapped_embed_model") is wrapped._wrapped_embed_model


def test_chunking_wrapper_async_query_embedding():
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    class AsyncEmbed:
        def __init__(self):
            self.called = []

        def get_text_embedding(self, text):
            self.called.append(text)
            return [0.3]

    wrapped = _wrap_chunking_embed_model(AsyncEmbed())

    assert asyncio.run(wrapped.aget_query_embedding("hello")) == [0.3]
    assert wrapped.called == ["hello"]


def test_preflight_batch_embedding_path(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import (
        _ensure_embedding_model_available,
    )

    embed_model = BatchOnlyEmbedModel()

    _ensure_embedding_model_available(embed_model)

    assert embed_model.calls == 1


def test_preflight_skips_when_no_embed_methods():
    from papairus.chat_with_repo.vector_store_manager import (
        _ensure_embedding_model_available,
    )

    class NoopEmbed:
        pass

    _ensure_embedding_model_available(NoopEmbed())


def test_missing_embedding_model_includes_base_url():
    from papairus.chat_with_repo.vector_store_manager import _raise_embedding_model_error

    embed_model = DummyEmbedModel(model_name="missing-embed", base_url="http://localhost:11434")

    with pytest.raises(MissingEmbeddingModelError) as excinfo:
        _raise_embedding_model_error(
            FakeResponseError("model \"missing-embed\" not found", status_code=404),
            embed_model,
        )

    assert "http://localhost:11434" in str(excinfo.value)


def test_raise_embedding_model_error_noop_for_unmatched():
    from papairus.chat_with_repo.vector_store_manager import _raise_embedding_model_error

    embed_model = DummyEmbedModel(model_name="dummy-embed", base_url="http://localhost:11434")

    # Should fall through without raising when the error is unrelated to embedding availability.
    _raise_embedding_model_error(FakeResponseError("bad request", status_code=400), embed_model)


def test_raise_embedding_model_error_handles_connection_markers():
    from papairus.chat_with_repo.vector_store_manager import _raise_embedding_model_error

    embed_model = DummyEmbedModel(model_name="dummy-embed", base_url="http://localhost:11434")

    with pytest.raises(EmbeddingServiceError):
        _raise_embedding_model_error(
            FakeResponseError("connection refused by host", status_code=None),
            embed_model,
        )


def test_missing_embedding_model_detected_by_message_only():
    from papairus.chat_with_repo.vector_store_manager import _raise_embedding_model_error

    embed_model = DummyEmbedModel(model_name="nomic-embed-text", base_url=None)

    with pytest.raises(MissingEmbeddingModelError):
        _raise_embedding_model_error(
            FakeResponseError("the specified model was not found", status_code=None),
            embed_model,
        )


def test_list_models_returns_none_when_request_fails(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _list_ollama_models

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert _list_ollama_models("http://localhost:11434") is None


def test_batch_embedding_failure_surfaces_base_url(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import (
        _ensure_embedding_model_available,
    )

    class FailingBatchEmbed(BatchOnlyEmbedModel):
        model_name = "missing-embed"
        base_url = "http://localhost:11434"

        def get_text_embedding_batch(self, texts):
            raise FakeResponseError("model \"missing-embed\" not found", status_code=404)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager._list_ollama_models",
        lambda _base_url: ["missing-embed"],
    )

    with pytest.raises(MissingEmbeddingModelError) as excinfo:
        _ensure_embedding_model_available(FailingBatchEmbed())

    assert "http://localhost:11434" in str(excinfo.value)


def test_preflight_reraises_non_embedding_errors():
    from papairus.chat_with_repo.vector_store_manager import (
        _ensure_embedding_model_available,
    )

    class ExplodingBatchEmbed(BatchOnlyEmbedModel):
        model_name = "unexpected"

        def get_text_embedding_batch(self, texts):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _ensure_embedding_model_available(ExplodingBatchEmbed())


def test_ollama_tag_listing_success(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _list_ollama_models

    class FakeResponse:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "a"}, {"name": "b"}]}

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.get",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    assert _list_ollama_models("http://localhost:11434") == ["a", "b"]


def test_model_normalization_handles_missing_name():
    from papairus.chat_with_repo.vector_store_manager import _normalize_model_name

    assert _normalize_model_name(None) is None


def test_ollama_tag_listing_skips_when_no_base_url():
    from papairus.chat_with_repo.vector_store_manager import _list_ollama_models

    assert _list_ollama_models(None) is None


def test_ollama_tag_listing_strips_latest(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _list_ollama_models

    class FakeResponse:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "nomic-embed-text:latest"}]}

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.get",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    assert _list_ollama_models("http://localhost:11434") == ["nomic-embed-text"]


def test_preflight_raises_when_tags_show_missing(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import (
        _ensure_embedding_model_available,
    )

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager._list_ollama_models",
        lambda _base_url: ["other-embed"],
    )

    embed_model = DummyEmbedModel(model_name="missing", base_url="http://localhost:11434")

    with pytest.raises(MissingEmbeddingModelError) as excinfo:
        _ensure_embedding_model_available(embed_model)

    assert "missing" in str(excinfo.value)


def test_preflight_accepts_tagged_model(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import (
        _ensure_embedding_model_available,
    )

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager._list_ollama_models",
        lambda _base_url: ["nomic-embed-text"],
    )

    embed_model = DummyEmbedModel(model_name="nomic-embed-text:latest", base_url="http://localhost:11434")

    # Should not raise since the normalized model name is present.
    _ensure_embedding_model_available(embed_model)


def test_preflight_raises_when_ollama_unreachable(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import (
        _ensure_embedding_model_available,
    )

    # Simulate unreachable Ollama host during tag listing
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager._list_ollama_models",
        lambda _base_url: None,
    )

    embed_model = DummyEmbedModel(
        model_name="nomic-embed-text", base_url="http://localhost:65535"
    )

    with pytest.raises(EmbeddingServiceError) as excinfo:
        _ensure_embedding_model_available(embed_model)

    assert "localhost:65535" in str(excinfo.value)


def test_preflight_surfaces_embedding_transport_error(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import (
        _ensure_embedding_model_available,
    )

    class ErroringEmbedModel(DummyEmbedModel):
        def get_text_embedding(self, _text):
            raise FakeResponseError(
                "do embedding request: Post \"http://127.0.0.1:64549/embedding\": EOF",
                status_code=500,
            )

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager._list_ollama_models",
        lambda _base_url: ["nomic-embed-text"],
    )

    embed_model = ErroringEmbedModel(
        model_name="nomic-embed-text", base_url="http://localhost:11434"
    )

    with pytest.raises(EmbeddingServiceError) as excinfo:
        _ensure_embedding_model_available(embed_model)

    assert "Embedding request to http://localhost:11434 failed" in str(excinfo.value)


