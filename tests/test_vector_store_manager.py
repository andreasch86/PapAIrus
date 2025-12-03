import types

import pytest

pytest.importorskip("chromadb")

from papairus.chat_with_repo.vector_store_manager import VectorStoreManager
from papairus.exceptions import MissingEmbeddingModelError


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
    manager = VectorStoreManager(top_k=1, llm="llm", embed_model="embed")
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
    patched_manager.embed_model = types.SimpleNamespace(
        model_name="missing-embed", base_url="http://localhost:11434"
    )

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
    patched_manager.embed_model = types.SimpleNamespace(model_name="missing-embed")

    def failing_index(*_args, **_kwargs):
        raise FakeResponseError("model \"missing-embed\" not found", status_code=404)

    monkeypatch.setattr(
        "papairus.chat_with_repo.vector_store_manager.VectorStoreIndex", failing_index
    )

    with pytest.raises(MissingEmbeddingModelError) as excinfo:
        patched_manager.create_vector_store(["content"], [{"source": "test"}])

    assert "pull missing-embed" in str(excinfo.value)


