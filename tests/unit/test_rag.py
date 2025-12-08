import types

import pytest

from papairus.chat_with_repo import rag as rag_module
from papairus.settings import ChatCompletionSettings


class DummyLLM:
    def __init__(self, complete_responses=None, chat_responses=None):
        self.complete_responses = list(complete_responses or [])
        self.chat_responses = list(chat_responses or [])

    def complete(self, prompt):
        if "search queries" in prompt:
            return types.SimpleNamespace(text="q1\nq2")
        if "Retrieve and Generate" in prompt or "Information:" in prompt:
            return types.SimpleNamespace(text="rag-text")
        if self.complete_responses:
            return types.SimpleNamespace(text=self.complete_responses.pop(0))
        return types.SimpleNamespace(text="default")

    def chat(self, *args, **kwargs):
        if args and args[0] == "rag-ar-prompt":
            return types.SimpleNamespace(message=types.SimpleNamespace(content="rag-ar-text"))
        content = self.chat_responses.pop(0) if self.chat_responses else "{}"
        return types.SimpleNamespace(message=types.SimpleNamespace(content=content))


class DummyTextAnalysisTool:
    def __init__(self, *_):
        self.calls = []

    def format_chat_prompt(self, message, instruction):
        self.calls.append(("format", message, instruction))
        return f"{instruction}:{message}"

    def keyword(self, prompt):
        self.calls.append(("keyword", prompt))
        return "keyword"

    def queryblock(self, message):
        return ["code-block"], [("md-block",)]

    def list_to_markdown(self, items):
        return "|".join(map(str, items))

    def nerquery(self, message):
        return f"ner-{message}"


class DummyVectorStoreManager:
    def __init__(self, top_k=None, llm=None, embed_model=None):
        self.created = False
        self.top_k = top_k
        self.llm = llm
        self.embed_model = embed_model

    def create_vector_store(self, *_):
        self.created = True

    def query_store(self, query):
        return [
            {"text": f"doc-{query}", "metadata": {"code_content": f"code-{query}"}},
            {"text": f"doc-{query}", "metadata": {"code_content": f"code-{query}-dup"}},
        ]


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    dummy_llm = DummyLLM(
        complete_responses=["q1\nq2", "rag-text", "rag-ar-text"],
        chat_responses=["rag-ar-text"],
    )
    monkeypatch.setattr(rag_module, "build_llm", lambda *_: dummy_llm)
    monkeypatch.setattr(rag_module, "build_embedding_model", lambda *_: "embed")
    monkeypatch.setattr(rag_module, "TextAnalysisTool", DummyTextAnalysisTool)
    monkeypatch.setattr(rag_module, "VectorStoreManager", DummyVectorStoreManager)
    monkeypatch.setattr(
        rag_module,
        "rag_ar_template",
        types.SimpleNamespace(format_messages=lambda **_: "rag-ar-prompt"),
    )
    return dummy_llm


def test_generate_queries_uses_llm(patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    queries = assistant.generate_queries("ask?", num_queries=3)

    assert queries == ["q1", "q2"]


def test_generate_queries_strips_formatting(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    monkeypatch.setattr(
        assistant.weak_model,
        "complete",
        lambda *_: types.SimpleNamespace(
            text=(
                "Sure, here are the two search queries you requested:\n\n"
                "**Query 1:**\n\n``""`\nSystem: What is this repo?\n``""`\n\n"
                "**Query 2:**\n\n``""`\nSystem: How does it work?\n``""`"
            )
        ),
    )

    queries = assistant.generate_queries("ask?", num_queries=3)

    assert queries == ["System: What is this repo?", "System: How does it work?"]


def test_sanitize_generated_queries_handles_non_strings(patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    cleaned = assistant._sanitize_generated_queries([None, 123, "   "])

    assert cleaned == []


def test_rerank_sorts_documents(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    docs = ["doc about apples", "query aligned doc"]
    results = assistant.rerank("query", docs)

    assert results == ["query aligned doc", "doc about apples"]


def test_rerank_handles_empty_query_terms(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    docs = ["doc-a", "doc-b", "doc-c"]
    results = assistant.rerank("!!!", docs)

    assert results == docs[:5]


def test_rerank_returns_empty_for_no_docs(patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    assert assistant.rerank("query", []) == []


def test_rag_and_rag_ar_delegate_to_llms(patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    assert assistant.rag("prompt", ["doc"]) == "rag-text"
    assert assistant.rag_ar("prompt", ["code"], ["doc"], "proj") == "rag-ar-text"


def test_respond_executes_full_flow(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    monkeypatch.setattr(assistant, "generate_queries", lambda *_args, **_kwargs: ["q1", "q2"])
    monkeypatch.setattr(assistant, "rerank", lambda _query, docs: list(reversed(docs)))
    monkeypatch.setattr(assistant, "rag", lambda *_: "ragged")
    monkeypatch.setattr(assistant, "rag_ar", lambda *_: "final")

    result = assistant.respond("question", "instruction")

    assert result[1] == "final"
    assert "doc-q2" in result[2]
    assert "code-q2" in result[4]


def test_respond_skips_empty_queries_and_returns_strings(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    seen_queries: list[str] = []

    monkeypatch.setattr(assistant, "generate_queries", lambda *_: ["", "q1", "   "])
    monkeypatch.setattr(assistant, "rag", lambda *_: "ragged")
    monkeypatch.setattr(assistant, "rag_ar", lambda *_: "final")
    monkeypatch.setattr(assistant, "rerank", lambda _query, docs: list(docs))
    monkeypatch.setattr(
        assistant.vector_store_manager,
        "query_store",
        lambda query: seen_queries.append(query) or [],
    )

    result = assistant.respond("question", "instruction")

    assert seen_queries == ["question", "q1"]
    assert isinstance(result[3], str)


def test_respond_deduplicates_queries(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    seen_queries: list[str] = []

    monkeypatch.setattr(assistant, "generate_queries", lambda *_: ["question", "other"])
    monkeypatch.setattr(assistant, "rag", lambda *_: "ragged")
    monkeypatch.setattr(assistant, "rag_ar", lambda *_: "final")
    monkeypatch.setattr(assistant, "rerank", lambda _query, docs: list(docs))
    monkeypatch.setattr(
        assistant.vector_store_manager,
        "query_store",
        lambda query: seen_queries.append(query) or [{"text": query, "metadata": {}}],
    )

    assistant.respond("question", "instruction")

    assert seen_queries == ["question", "other"]


def test_respond_flattens_markdown_variants(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    class MixedTextAnalysisTool(DummyTextAnalysisTool):
        def queryblock(self, _message):
            return ["code-a"], [["md-list", None], ("tuple-md", ""), "string-md"]

        def list_to_markdown(self, items):
            return "|".join(map(str, items))

        def nerquery(self, message):
            return message

    assistant.textanslys = MixedTextAnalysisTool(None, None)
    assistant.rerank = lambda _query, docs: list(docs)
    assistant.rag = lambda *_: "ragged"
    captured_embedding: list[str] = []

    def _capture_rag_ar(_prompt, _code, embedding_recall, _project):
        captured_embedding.extend(embedding_recall)
        return "final"

    assistant.rag_ar = _capture_rag_ar
    assistant.generate_queries = lambda *_: []
    assistant.vector_store_manager.query_store = lambda *_: [
        {"text": "doc-one", "metadata": {"code_content": "code-one"}}
    ]

    result = assistant.respond("question", "instruction")

    assert "code-a" in result[5]
    assert any("md-list" in item for item in captured_embedding)


def test_respond_flattens_string_only_metadata(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    class StringOnlyTool(DummyTextAnalysisTool):
        def queryblock(self, _message):
            return ["code-b"], ["", "only-string-md", 123]

        def list_to_markdown(self, items):
            return "|".join(map(str, items))

        def nerquery(self, message):
            return message

    assistant.textanslys = StringOnlyTool(None, None)
    assistant.rerank = lambda _query, docs: list(docs)
    assistant.rag = lambda *_: "ragged"
    assistant.rag_ar = lambda *_: "final"
    assistant.generate_queries = lambda *_: []
    assistant.vector_store_manager.query_store = lambda *_: [
        {"text": "doc-two", "metadata": {"code_content": "code-two"}}
    ]

    result = assistant.respond("question", "instruction")

    assert "code-b" in result[5]


def test_respond_falls_back_to_message_when_no_queries(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    seen_queries: list[str] = []

    monkeypatch.setattr(assistant, "generate_queries", lambda *_: [])
    monkeypatch.setattr(assistant, "rag", lambda *_: "ragged")
    monkeypatch.setattr(assistant, "rag_ar", lambda *_: "final")
    monkeypatch.setattr(assistant, "rerank", lambda _query, docs: list(docs))
    monkeypatch.setattr(
        assistant.vector_store_manager,
        "query_store",
        lambda query: seen_queries.append(query) or [],
    )

    assistant.respond("question", "instruction")

    assert seen_queries == ["question"]


def test_respond_returns_fallback_when_no_results(monkeypatch, patch_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="local-gemma")
    assistant = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    monkeypatch.setattr(assistant.vector_store_manager, "query_store", lambda *_: [])

    result = assistant.respond("question", "instruction")

    assert "could not find any relevant information" in result[1]
    assert result[2] == ""
