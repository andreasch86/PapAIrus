from pathlib import Path

from llama_index.core.llms import ChatMessage

from papairus.docstring_generator import DocstringGenerator


def test_adds_missing_docstring(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("def add(a: int, b: int):\n    return a + b\n")

    generator = DocstringGenerator(tmp_path)
    updated = generator.run()

    content = sample.read_text()
    assert sample in updated
    assert "Args:" in content
    assert "a (int)" in content and "b (int)" in content
    assert "Returns:" in content


def test_updates_incomplete_docstring(tmp_path):
    sample = tmp_path / "greet.py"
    sample.write_text(
        "def greet(name, greeting='hi'):\n"
        "    \"\"\"Greet the user.\n\n"
        "    Args:\n"
        "        name (str): Name of the user.\n"
        "    \"\"\"\n"
        "    return f\"{greeting}, {name}\"\n"
    )

    generator = DocstringGenerator(tmp_path)
    updated = generator.run()

    content = sample.read_text()
    assert sample in updated
    assert "greeting" in content
    assert "Returns:" in content


def test_skips_init_methods(tmp_path):
    sample = tmp_path / "init_sample.py"
    sample.write_text(
        "class Thing:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n"
    )

    generator = DocstringGenerator(tmp_path)
    updated = generator.run()

    content = sample.read_text()
    assert sample in updated
    assert 'def __init__(self, value):\n        """' not in content


def test_llm_backend_uses_client(tmp_path):
    sample = tmp_path / "llm_sample.py"
    sample.write_text("def shout(text):\n    return text.upper()\n")

    class FakeLLM:
        def chat(self, messages):
            return type("Resp", (), {"message": type("Msg", (), {"content": '"""Shout text.\n\nArgs:\n    text (str): Input.\nReturns:\n    str: Uppercase.\n"""'})()})()

    generator = DocstringGenerator(tmp_path, backend="gemini", llm_client=FakeLLM())
    updated = generator.run()

    assert sample in updated
    content = sample.read_text()
    assert "Shout text." in content
    assert "Args:" in content and "Returns:" in content


def test_llm_backend_uses_chat_messages(tmp_path):
    sample = tmp_path / "llm_chatmessage.py"
    sample.write_text("def ping(x):\n    return x\n")

    class RecordingLLM:
        def __init__(self):
            self.messages = None

        def chat(self, messages):
            self.messages = messages
            return type(
                "Resp",
                (),
                {
                    "message": type(
                        "Msg",
                        (),
                        {
                            "content": '"""Ping value.\n\nArgs:\n    x (Any): Description of x.\nReturns:\n    Any: Description of return value.\n"""',
                        },
                    )(),
                },
            )()

    llm = RecordingLLM()
    generator = DocstringGenerator(tmp_path, backend="gemma", llm_client=llm)
    updated = generator.run()

    assert sample in updated
    assert llm.messages and isinstance(llm.messages[0], ChatMessage)
    assert "Ping value." in sample.read_text()


def test_progress_callback_reports_status(tmp_path):
    sample = tmp_path / "progress_sample.py"
    sample.write_text("def hello(name):\n    return name\n")

    statuses = []

    def record_progress(path, status):
        statuses.append((path.name, status))

    generator = DocstringGenerator(tmp_path)
    generator.run(progress_callback=record_progress)

    assert ("progress_sample.py", "start") in statuses
    assert any(status == "updated" for _, status in statuses)
