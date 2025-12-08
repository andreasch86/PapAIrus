import types

import pytest

from papairus.chat_with_repo import rag as rag_module
from papairus.settings import ChatCompletionSettings


class RecordingVectorStoreManager:
    def __init__(self, *_args, **_kwargs):
        self.queries: list[str] = []

    def query_store(self, query: str):
        self.queries.append(query)
        return [{"text": f"doc-{query}", "metadata": {"code_content": f"code-{query}"}}]


class FakeLLM:
    def complete(self, prompt):
        if "Please provide a list of Code keywords" in prompt:
            return types.SimpleNamespace(text="keyword")

        if "Generate" in prompt or "search queries" in prompt:
            return types.SimpleNamespace(
                text=(
                    "Sure, here are the two search queries you requested:\n\n"
                    "**Query 1:**\n\n``\nSystem: What is the purpose of this repository?\n``\n\n"
                    "**Query 2:**\n\n``\nSystem: How does it work?\n```"
                )
            )

        return types.SimpleNamespace(text="default-complete")

    def chat(self, *_args, **_kwargs):
        return types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))


class StubTextAnalysisTool:
    """
    A tool for analyzing text.
    
    Args:
        *_: Unused positional arguments.
    """
    def __init__(self, *_):
        pass

    def format_chat_prompt(self, message, instruction):
        """
        Formats a chat prompt.
        
        Args:
            message: The message to format.
            instruction: The instruction to format.
        
        Returns:
            The formatted prompt.
        """
        return f"System:{instruction}\nUser:{message}\nAssistant:"

    def keyword(self, prompt):
        """
        Returns a keyword.
        
        Args:
            prompt: The prompt to generate the keyword from.
        
        Returns:
            A SimpleNamespace object with the keyword text.
        """
        return types.SimpleNamespace(text="kw")

    def queryblock(self, _message):
        """
        Returns the query block.
        
        Args:
            message: The message to generate the query block from.
        
        Returns:
            A list of code block and markdown block types.
        """
        return ["code-block"], ["md-block"]

    def list_to_markdown(self, items):
        """
        Converts a list to markdown.
        
        Args:
            items: The list to convert.
        
        Returns:
            The markdown string.
        """
        return "\n".join(map(str, items))

    def nerquery(self, message):
        return message


@pytest.fixture()
def patched_dependencies(monkeypatch):
    monkeypatch.setattr(rag_module, "build_llm", lambda *_: FakeLLM())
    monkeypatch.setattr(rag_module, "build_embedding_model", lambda *_: "embed")
    monkeypatch.setattr(rag_module, "VectorStoreManager", RecordingVectorStoreManager)
    monkeypatch.setattr(rag_module, "TextAnalysisTool", StubTextAnalysisTool)
    monkeypatch.setattr(
        rag_module,
        "rag_ar_template",
        types.SimpleNamespace(format_messages=lambda **_: "rag-ar-prompt"),
    )


@pytest.fixture()
def assistant(patched_dependencies, tmp_path):
    settings = ChatCompletionSettings(model="codegemma")
    instance = rag_module.RepoAssistant(settings, tmp_path / "db.json")

    # Simplify downstream LLM usage to keep the test focused on query handling
    instance.rerank = lambda _query, docs: list(docs)
    instance.rag = lambda *_: "ragged"
    instance.rag_ar = lambda *_: "final"

    return instance


def test_respond_uses_clean_queries_and_skips_blanks(assistant):
    response = assistant.respond("question", "instruction")

    assert response[1] == "final"
    assert assistant.vector_store_manager.queries == [
        "question",
        "System: What is the purpose of this repository?",
        "System: How does it work?",
    ]
