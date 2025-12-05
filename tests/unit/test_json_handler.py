import json

import pytest

from papairus.chat_with_repo.json_handler import JsonFileProcessor


def test_extract_data_collects_md_and_metadata(tmp_path):
    payload = {
        "file.py": [
            {
                "md_content": ["md body"],
                "type": "function",
                "name": "do_work",
                "code_start_line": 1,
                "code_end_line": 2,
                "have_return": True,
                "code_content": "def do_work(): pass",
                "name_column": 3,
                "item_status": "active",
            }
        ]
    }
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))

    processor = JsonFileProcessor(json_path)
    md_contents, extracted = processor.extract_data()

    assert md_contents == ["md body"]
    assert extracted == [
        {
            "type": "function",
            "name": "do_work",
            "code_start_line": 1,
            "code_end_line": 2,
            "have_return": True,
            "code_content": "def do_work(): pass",
            "name_column": 3,
            "item_status": "active",
        }
    ]


def test_read_json_file_missing_exits(tmp_path):
    processor = JsonFileProcessor(tmp_path / "missing.json")

    with pytest.raises(SystemExit):
        processor.read_json_file()


def test_search_code_contents_by_name_success(tmp_path):
    payload = {"files": [{"name": "target", "code_content": "code", "md_content": ["md"]}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))

    processor = JsonFileProcessor(json_path)
    codes, mds = processor.search_code_contents_by_name(json_path, "target")

    assert codes == ["code"]
    assert mds == [["md"]]


def test_search_code_contents_by_name_handles_errors(tmp_path):
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not-json")
    processor = JsonFileProcessor(invalid_path)

    assert processor.search_code_contents_by_name(invalid_path, "anything") == "Invalid JSON file."

    missing_path = tmp_path / "missing.json"
    assert processor.search_code_contents_by_name(missing_path, "anything") == "File not found."


def test_extract_data_handles_non_list_and_missing_content(tmp_path):
    payload = {"file.py": {"unexpected": "structure"}, "other.py": [{"name": "noop"}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))

    processor = JsonFileProcessor(json_path)
    md_contents, extracted = processor.extract_data()

    assert md_contents == []
    assert extracted == []


def test_search_code_contents_by_name_no_match_and_exception(monkeypatch, tmp_path):
    payload = {"files": [{"name": "target", "code_content": "code", "md_content": ["md"]}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))
    processor = JsonFileProcessor(json_path)

    no_match_codes, no_match_mds = processor.search_code_contents_by_name(json_path, "missing")
    assert no_match_codes == ["No matching item found."]
    assert no_match_mds == ["No matching item found."]

    def boom(*_, **__):
        raise RuntimeError("boom")

    monkeypatch.setattr(json, "load", boom)
    assert processor.search_code_contents_by_name(json_path, "target") == "An error occurred: boom"


def test_search_code_contents_by_name_missing_code_content(tmp_path):
    payload = {"files": [{"name": "target", "md_content": ["md"]}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))
    processor = JsonFileProcessor(json_path)

    codes, mds = processor.search_code_contents_by_name(json_path, "target")

    assert codes == ["No matching item found."]
    assert mds == ["No matching item found."]
