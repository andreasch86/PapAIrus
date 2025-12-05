from pathlib import Path

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
