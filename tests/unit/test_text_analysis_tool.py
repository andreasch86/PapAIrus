import types

from papairus.chat_with_repo.text_analysis_tool import TextAnalysisTool


class DummyLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return types.SimpleNamespace(text="done")


def test_text_analysis_tool_invokes_llm(monkeypatch, tmp_path):
    dummy_llm = DummyLLM()
    tool = TextAnalysisTool(dummy_llm, tmp_path / "db.json")

    assert tool.keyword("hello").text == "done"
    assert tool.tree("tree").text == "done"
    assert "System:instructions" not in tool.format_chat_prompt("a", "b")

    prompt = tool.format_chat_prompt("a", "b")
    assert prompt == "System:b\nUser: a\nAssistant:"

    monkeypatch.setattr(
        tool.jsonsearch, "search_code_contents_by_name", lambda path, message: (["code"], [["md"]])
    )
    assert tool.queryblock("needle") == (["code"], [["md"]])

    markdown = tool.list_to_markdown(["one", "two"])
    assert "1. one" in markdown and "2. two" in markdown

    tool.nerquery("what")
    assert any("The input is shown" in prompt for prompt in dummy_llm.prompts)
