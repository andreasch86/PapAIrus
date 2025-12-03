import types

import pytest

pytest.importorskip("chromadb")

from papairus.chat_with_repo.vector_store_manager import VectorStoreManager


class DummyDocument:
    def __init__(self, text, extra_info=None):
        self.text = text
        self.extra_info = extra_info or {}


class DummyDocumentNoText:
    def __init__(self, extra_info=None):
        self.extra_info = extra_info or {}


class DummySplitter:
    def __init__(self, *args, **kwargs):
        pass

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


