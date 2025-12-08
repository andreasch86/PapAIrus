import asyncio
import sys
import types

import pytest

pytest.importorskip("chromadb")

from papairus.chat_with_repo.vector_store_manager import (
    _MAX_EMBED_CHARS,
    VectorStoreManager,
    _ChunkingEmbeddingWrapper,
    _apply_system_prompt,
    _build_repo_system_prompt,
    _get_node_content,
    _rechunk_oversized_nodes,
)
from papairus.exceptions import EmbeddingServiceError, MissingEmbeddingModelError


@pytest.fixture(autouse=True)
def prevent_real_ollama(monkeypatch):
    """Avoid contacting a real Ollama daemon during unit tests."""

    monkeypatch.setitem(sys.modules, "ollama", None)


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


class DummyCollection:
    def __init__(self, name):
        self.name = name

    def get(self, *args, **kwargs):
        return {"ids": []}


class DummyPersistentClient:
    def __init__(self, path):
        self.path = path

    def get_or_create_collection(self, name):
        return DummyCollection(name)


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


class RecordingLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def update_system_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)


class DummyEmbedModel:
    def __init__(self, *, model_name="dummy-embed", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def get_text_embedding(self, _text):
        return [0.1, 0.2]


class OversizedNode:
    def __init__(self, text, meta=None):
        self.text = text
        self.extra_info = meta or {"source": "demo"}

    def get_content(self, metadata_mode=None):
        return self.text


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


def test_create_vector_store_filters_empty_chunks(monkeypatch, patched_manager):
    class EmptyNode:
        def get_content(self, metadata_mode=None):
            return "   "

    class MixedSplitter:
        def __init__(self, *args, **kwargs):
            self.chunk_size = kwargs.get("chunk_size")

        def get_nodes_from_documents(self, docs):
            return [EmptyNode(), "node-keep"]

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SemanticSplitterNodeParser",
        MixedSplitter,
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SentenceSplitter",
        MixedSplitter,
    )

    patched_manager.create_vector_store(["content"], [{"source": "test"}])

    assert patched_manager.query_engine is not None
    assert patched_manager.query_engine.retriever.index.nodes == ["node-keep"]


def test_system_prompt_applied_from_metadata(patched_manager):
    patched_manager.llm = RecordingLLM()

    md_contents = ["hello world"]
    meta_data = [{"source": "src/app.py", "type": "Function", "name": "foo"}]

    patched_manager.create_vector_store(md_contents, meta_data)

    assert patched_manager.llm.prompts
    prompt = patched_manager.llm.prompts[-1]
    assert "src/app.py" in prompt
    assert "Function" in prompt


def test_create_vector_store_uses_thread_pool(monkeypatch, patched_manager):
    captured = {}

    class RecordingFuture:
        def __init__(self, payload):
            self._payload = payload

        def result(self):
            return self._payload

        def __hash__(self):
            return id(self)

    class RecordingExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers
            self._futures = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            result = fn(*args, **kwargs)
            future = RecordingFuture(result)
            self._futures.append(future)
            return future

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.ThreadPoolExecutor", RecordingExecutor
    )
    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.as_completed", lambda futures: futures
    )

    patched_manager.max_workers = 8

    md_contents = ["doc1", "doc2", "doc3"]
    meta_data = [{"source": "a"}, {"source": "b"}, {"source": "c"}]

    patched_manager.create_vector_store(md_contents, meta_data)

    assert captured["max_workers"] == len(md_contents)


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


def test_get_node_content_falls_back_to_text():
    node = OversizedNode("node-text")

    class TextOnly:
        """
        A class that represents text-only content.
        
        Attributes:
            text: The text content.
        """
        text = "text-only"

    class Raises:
        """
        A class that raises an error when getting content.
        
        Methods:
            get_content(self, metadata_mode=None): Raises a RuntimeError.
        """
        def get_content(self, metadata_mode=None):
            """
            A method that raises an error when getting content.
            
            Args:
                self: The instance of the class.
                metadata_mode: The metadata mode.
            
            Raises:
                RuntimeError: An error occurred.
            """
            raise RuntimeError("boom")

    assert _get_node_content(node) == "node-text"
    assert _get_node_content(TextOnly()) == "text-only"
    assert _get_node_content(Raises()) == ""


def test_create_vector_store_skips_when_missing_data(patched_manager):
    """
    A test that verifies that the vector store is not created when missing data.
    
    Args:
        patched_manager: A patched vector store manager.
    """
    patched_manager.create_vector_store([], [])

    assert patched_manager.query_engine is None


def test_create_vector_store_handles_empty_nodes(monkeypatch, patched_manager):
    """
    A test that verifies that the vector store is not created when empty nodes are provided.
    
    Args:
        monkeypatch: A pytest monkeypatch.
        patched_manager: A patched vector store manager.
    """
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


def test_rechunk_oversized_nodes(monkeypatch):
    big_text = "x" * (_MAX_EMBED_CHARS + 10)
    oversized = OversizedNode(big_text, {"source": "demo"})
    small = OversizedNode("ok")

    class LocalSentenceSplitter:
        def __init__(self, chunk_size, chunk_overlap=0):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def get_nodes_from_documents(self, docs):
            text = docs[0].text
            mid = len(text) // 2
            return [OversizedNode(text[:mid], docs[0].extra_info), OversizedNode(text[mid:], docs[0].extra_info)]

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SentenceSplitter",
        LocalSentenceSplitter,
    )

    balanced = _rechunk_oversized_nodes([oversized, small], max_chars=100)

    assert len(balanced) >= 3
    assert all(len(_get_node_content(node)) <= 100 for node in balanced)
    assert any(node.extra_info == {"source": "demo"} for node in balanced)


def test_rechunk_oversized_nodes_respects_metadata(monkeypatch):
    big_text = "x" * (_MAX_EMBED_CHARS + 50)
    meta = {"source": "demo", "notes": "y" * 5000}
    seen_chunk_sizes = []

    class CapturingSentenceSplitter:
        def __init__(self, chunk_size, chunk_overlap=0):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            seen_chunk_sizes.append(chunk_size)

        def get_nodes_from_documents(self, docs):
            text = docs[0].text
            mid = len(text) // 2
            return [
                OversizedNode(text[:mid], docs[0].extra_info),
                OversizedNode(text[mid:], docs[0].extra_info),
            ]

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SentenceSplitter",
        CapturingSentenceSplitter,
    )

    balanced = _rechunk_oversized_nodes([OversizedNode(big_text, meta)], max_chars=100)

    assert seen_chunk_sizes and seen_chunk_sizes[0] >= len(str(meta)) + 1
    assert all(node.extra_info == meta for node in balanced)
    assert all(len(_get_node_content(node)) <= seen_chunk_sizes[0] for node in balanced)


def test_rechunk_oversized_nodes_handles_no_progress(monkeypatch):
    big_text = "y" * 250
    oversized = OversizedNode(big_text, {"source": "stuck"})

    class NoProgressSplitter:
        def __init__(self, chunk_size, chunk_overlap=0):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def get_nodes_from_documents(self, docs):
            # Always return the same oversized node to simulate a splitter that cannot
            # reduce the size (e.g., when text lacks separators).
            return [OversizedNode(docs[0].text, docs[0].extra_info)]

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.SentenceSplitter",
        NoProgressSplitter,
    )

    balanced = _rechunk_oversized_nodes([oversized], max_chars=100)

    assert len(balanced) == 3
    assert all(len(_get_node_content(node)) <= 100 for node in balanced)
    assert all(node.extra_info == {"source": "stuck"} for node in balanced)


def test_rechunk_oversized_nodes_requires_positive_size():
    with pytest.raises(ValueError):
        _rechunk_oversized_nodes([OversizedNode("abc")], max_chars=0)


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


def test_chunking_wrapper_http_fallback(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    class FailingEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

        def get_text_embedding(self, _text):
            raise FakeResponseError("embed failed", status_code=500)

    def fake_post(url, json, timeout):
        assert url.endswith("/api/embeddings")
        assert json["model"] == "nomic-embed-text"
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": [0.1, 0.2]}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapped = _wrap_chunking_embed_model(FailingEmbed())

    assert wrapped.get_text_embedding("hello") == [0.1, 0.2]
    assert wrapped.get_text_embedding_batch(["hello", "world"]) == [
        [0.1, 0.2],
        [0.1, 0.2],
    ]


def test_chunking_wrapper_http_fallback_for_batch(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    class BatchFailureEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

        def get_text_embedding_batch(self, _texts):
            raise FakeResponseError("batch failed", status_code=500)

    def fake_post(url, json, timeout):
        assert url.endswith("/api/embeddings")
        assert json["model"] == "nomic-embed-text"

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"embedding": [0.3, 0.4]}]}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapped = _wrap_chunking_embed_model(BatchFailureEmbed())

    assert wrapped.get_text_embedding_batch(["a", "b"]) == [[0.3, 0.4], [0.3, 0.4]]


def test_embed_via_http_requires_base_url(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    wrapper = _ChunkingEmbeddingWrapper(object(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_embed_via_http_prefers_ollama_client(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    class FakeClient:
        def __init__(self, host):
            assert host == "http://localhost:11434"

        def embeddings(self, **kwargs):
            assert kwargs == {"model": "nomic-embed-text", "prompt": "text"}
            return {"embedding": [0.42, 0.43]}

    fake_ollama = type("FakeOllama", (), {"Client": FakeClient})
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    def fake_post(_url, _json, _timeout):
        raise AssertionError("HTTP fallback should not be hit when client succeeds")

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    assert wrapper._embed_via_http("text") == [0.42, 0.43]


def test_chunking_wrapper_allows_private_setattr():
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)
    wrapper._embed_batch_via_http = lambda texts: [[1.0] for _ in texts]  # type: ignore[attr-defined]

    assert wrapper._embed_batch_via_http(["a"]) == [[1.0]]


def test_chunking_wrapper_allows_public_setattr():
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)
    wrapper.model_name = "different-model"

    assert wrapper.model_name == "different-model"


def test_embed_via_http_skips_client_without_embedding(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    class FakeClient:
        def __init__(self, host):
            assert host == "http://localhost:11434"

        def embeddings(self, **kwargs):
            return {"unexpected": True}

    fake_ollama = type("FakeOllama", (), {"Client": FakeClient})
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    called = {}

    def fake_post(url, json, timeout):
        called["hit"] = True

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": [0.7, 0.8]}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    assert wrapper._embed_via_http("text") == [0.7, 0.8]
    assert called["hit"] is True


def test_embed_via_http_raises_on_bad_payload(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"unexpected": True}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_embed_via_http_supports_embeddings_key(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [[0.9, 1.0]]}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    assert wrapper._embed_via_http("text") == [0.9, 1.0]


def test_embed_via_http_handles_non_dict(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return [1, 2, 3]

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_embed_via_http_retries_with_input_on_500(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    call_count = {"value": 0}

    def fake_post(url, json, timeout):
        assert url.endswith("/api/embeddings")
        call_count["value"] += 1

        class Response:
            def __init__(self, status_code):
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code == 500:
                    from requests import HTTPError

                    http_error = HTTPError()
                    http_error.response = self
                    raise http_error
                return None

            def json(self):
                return {"embedding": [0.5, 0.6]}

        # First call returns server error to trigger fallback to the "input" payload.
        if call_count["value"] == 1:
            return Response(500)
        return Response(200)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    assert wrapper._embed_via_http("text") == [0.5, 0.6]
    assert call_count["value"] == 2


def test_embed_via_http_raises_after_double_500(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            status_code = 500

            def raise_for_status(self):
                from requests import HTTPError

                error = HTTPError()
                error.response = self
                raise error

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_embed_via_http_raises_on_client_error(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            status_code = 404
            text = "model not found"

            def raise_for_status(self):
                from requests import HTTPError

                error = HTTPError(self.text)
                error.response = self
                raise error

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(MissingEmbeddingModelError):
        wrapper._embed_via_http("text")


def test_embed_via_http_rejects_bad_embeddings_payload(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": ["oops"]}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_embed_via_http_requires_embedding_in_data(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"missing": True}]}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_embed_via_http_rejects_empty_embedding(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": []}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_embed_via_http_retries_with_alternate_payload_on_empty(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                # First payload (prompt) returns an empty embedding; second (input)
                # returns a valid vector.
                if "prompt" in self._payload:
                    return {"embedding": []}
                return {"embedding": [0.1, 0.2]}

        return Response(json)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    assert wrapper._embed_via_http("text") == [0.1, 0.2]


def test_embed_via_http_substitutes_zero_vector_on_empty_payload(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": []}

        return Response(json)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)
    wrapper._last_embedding_dim = 2

    assert wrapper._embed_via_http("text") == [0.0, 0.0]


def test_embed_via_http_rejects_non_list_embedding(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": {"bad": True}}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_embed_via_http_rejects_non_numeric_embedding(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BareEmbed:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

    def fake_post(url, json, timeout):
        """
        Generates a fake response object for testing purposes.
        
        Args:
            url: The URL to send the request to.
            json: The JSON data to send with the request.
            timeout: The timeout for the request.
        
        Returns:
            A Response object with a mocked response.
        """
        class Response:
            """
            A class representing a response object.
            
            Attributes:
                None
            
            Methods:
                raise_for_status(): Raises an exception if the response status code is not 200.
                json(): Returns the response data as a dictionary.
            """
            def raise_for_status(self):
                """
                Raises an exception if the response status code is not 200.
                
                Args:
                    self: The Response object.
                
                Returns:
                    None
                """
                return None

            def json(self):
                """
                Returns the response data as a dictionary.
                
                Args:
                    self: The Response object.
                
                Returns:
                    A dictionary containing the response data.
                """
                return {"embedding": [1, "bad"]}

        return Response()

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.requests.post", fake_post
    )

    wrapper = _ChunkingEmbeddingWrapper(BareEmbed(), max_batch_size=1)

    with pytest.raises(EmbeddingServiceError):
        wrapper._embed_via_http("text")


def test_chunking_wrapper_get_text_embedding_missing_method():
    """
    Tests that the `_wrap_chunking_embed_model()` function raises an exception if the wrapped object does not have a `get_text_embedding()` method.
    
    Args:
        None
    
    Returns:
        None
    """
    from papairus.chat_with_repo.vector_store_manager import (
        _wrap_chunking_embed_model,
    )

    class BatchOnly:
        def get_text_embedding_batch(self, texts):
            return [[1.0] for _ in texts]

    wrapped = _wrap_chunking_embed_model(BatchOnly())

    with pytest.raises(AttributeError):
        wrapped.get_text_embedding("nope")


def test_chunking_wrapper_rejects_empty_batch_embedding():
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BatchOnly:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

        def get_text_embedding_batch(self, texts):
            return [[] for _ in texts]

    wrapped = _ChunkingEmbeddingWrapper(BatchOnly())

    with pytest.raises(EmbeddingServiceError):
        wrapped.get_text_embedding_batch(["text"])


def test_chunking_wrapper_rejects_mismatched_batch_length():
    """
    Test that the chunking wrapper rejects batches with mismatched lengths.
    
    This test simulates a service bug where the embedding service returns fewer embeddings than requested. The chunking wrapper should raise an `EmbeddingServiceError` in this case.
    """
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BatchOnly:
        """
        A class that provides text embedding functionality.
        
        This class simulates an external service that only provides embeddings for batches of text.
        """
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

        def get_text_embedding_batch(self, texts):
            # Return fewer embeddings than requested to simulate a service bug.
            """
            Get text embeddings for a batch of texts.
            
            This method simulates a service bug where the embedding service returns fewer embeddings than requested. The chunking wrapper should raise an `EmbeddingServiceError` in this case.
            
            Args:
                texts: A list of texts to embed.
            
            Returns:
                A list of text embeddings.
            """
            return [[1.0]]

    wrapped = _ChunkingEmbeddingWrapper(BatchOnly(), max_batch_size=2)
    wrapped._embed_batch_via_http = lambda batch: []  # type: ignore[attr-defined]

    with pytest.raises(EmbeddingServiceError):
        wrapped.get_text_embedding_batch(["a", "b"])


def test_chunking_wrapper_ignores_show_progress_kwarg():
    """
    Test that the chunking wrapper ignores the `show_progress` kwarg.
    
    This test ensures that the chunking wrapper does not raise an error if the `show_progress` kwarg is set to `False`.
    """
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BatchOnly:
        """
        A class that provides text embedding functionality.
        
        This class simulates an external service that only provides embeddings for batches of text.
        """
        model_name = "nomic-embed-text"

        def get_text_embedding_batch(self, texts):
            return [[float(idx)] for idx, _ in enumerate(texts)]

    wrapped = _ChunkingEmbeddingWrapper(BatchOnly(), max_batch_size=2)

    embeddings = wrapped.get_text_embedding_batch(["a", "b"], show_progress=False)

    assert embeddings == [[0.0], [1.0]]


def test_chunking_wrapper_http_branch_raises_on_empty(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BatchRaises:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

        def get_text_embedding_batch(self, texts):
            raise RuntimeError("boom")

    wrapped = _ChunkingEmbeddingWrapper(BatchRaises(), max_batch_size=2)
    monkeypatch.setattr(wrapped, "_embed_batch_via_http", lambda batch: [])

    with pytest.raises(EmbeddingServiceError):
        wrapped.get_text_embedding_batch(["only-one"])


def test_chunking_wrapper_recovers_after_http_retry(monkeypatch):
    from papairus.chat_with_repo.vector_store_manager import _ChunkingEmbeddingWrapper

    class BatchMismatch:
        base_url = "http://localhost:11434"
        model_name = "nomic-embed-text"

        def get_text_embedding_batch(self, texts):
            return []

    wrapped = _ChunkingEmbeddingWrapper(BatchMismatch(), max_batch_size=4)
    monkeypatch.setattr(
        wrapped,
        "_embed_batch_via_http",
        lambda batch: [[0.1] for _ in batch],
    )

    assert wrapped.get_text_embedding_batch(["a", "b"]) == [[0.1], [0.1]]


def test_get_node_content_unknown_type_returns_empty():
    class Unknown:
        pass

    assert _get_node_content(Unknown()) == ""


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


def test_build_repository_system_prompt_includes_examples():
    meta = [
        {"name": "file.py", "type": "File", "source": "repo/file.py"},
        {"name": "function", "type": "FunctionDef"},
    ]

    prompt = _build_repo_system_prompt(meta)

    assert "Examples of indexed items" in prompt
    assert "from repo/file.py" in prompt


def test_build_repository_system_prompt_handles_empty_examples():
    prompt = _build_repo_system_prompt([])

    assert "Examples of indexed items" not in prompt


def test_apply_system_prompt_fallback_sets_dict():
    class FailingLLM:
        def update_system_prompt(self, _prompt):  # pragma: no cover - exercised via call
            raise RuntimeError("fail")

        @property
        def system_prompt(self):  # pragma: no cover - getter not used
            return ""

        @system_prompt.setter
        def system_prompt(self, _value):
            raise RuntimeError("nope")

    llm = FailingLLM()
    _apply_system_prompt(llm, "context-prompt")

    assert llm.__dict__["system_prompt"] == "context-prompt"


def test_apply_system_prompt_sets_attribute_directly():
    class SimpleLLM:
        def __init__(self):
            self.system_prompt = ""

    llm = SimpleLLM()
    _apply_system_prompt(llm, "direct-prompt")

    assert llm.system_prompt == "direct-prompt"


def test_chunking_wrapper_get_query_embedding_handles_wrapped():
    class QueryEmbedModel(DummyEmbedModel):
        def get_query_embedding(self, query):
            return [len(query)]

    wrapper = _ChunkingEmbeddingWrapper(QueryEmbedModel())

    assert wrapper.get_query_embedding("hi") == [0.1, 0.2]


def test_chunking_wrapper_http_error_raises(monkeypatch):
    class BatchModel:
        def get_text_embedding_batch(self, _batch):
            raise RuntimeError("batch boom")

    wrapper = _ChunkingEmbeddingWrapper(BatchModel())
    monkeypatch.setattr(
        wrapper, "_embed_batch_via_http", lambda *_: (_ for _ in ()).throw(ValueError("http fail"))
    )

    with pytest.raises(EmbeddingServiceError):
        wrapper._get_text_embeddings(["a", "b"])


def test_chunking_wrapper_recovers_from_base_init_typeerror(monkeypatch):
    import papairus.chat_with_repo.vector_store_manager as vsm

    original_init = vsm.BaseEmbedding.__init__

    def flaky_init(self, *args, **kwargs):
        fail_once = getattr(self, "_fail_once", True)
        if fail_once:
            self._fail_once = False
            raise TypeError("boom")
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(vsm.BaseEmbedding, "__init__", flaky_init)

    wrapper = _ChunkingEmbeddingWrapper(DummyEmbedModel())

    assert wrapper._wrapped_embed_model.model_name == "dummy-embed"


def test_create_vector_store_raises_on_process_failure(monkeypatch, patched_manager):
    patched_manager._process_document = lambda *_args: (_ for _ in ()).throw(ValueError("split fail"))

    with pytest.raises(ValueError):
        patched_manager.create_vector_store(["text"], [{}])


