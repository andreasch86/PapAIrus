import json

import pytest

from papairus.chat_with_repo.json_handler import JsonFileProcessor


def test_extract_data_collects_md_and_metadata(tmp_path):
    """
    Test case for the `test_extract_data_collects_md_and_metadata` function.
    
    This function tests that the `extract_data` method of the `JsonFileProcessor` class correctly extracts the markdown content and extracted data from a valid JSON file.
    
    Args:
        tmp_path: A temporary directory path.
    
    Returns:
        None.
    """
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
    """
    Test case for the `test_read_json_file_missing_exits` function.
    
    This function tests that the `read_json_file` method of the `JsonFileProcessor` class raises a `SystemExit` exception when the JSON file does not exist.
    
    Args:
        tmp_path: A temporary directory path.
    
    Returns:
        None.
    """
    processor = JsonFileProcessor(tmp_path / "missing.json")

    with pytest.raises(SystemExit):
        processor.read_json_file()


def test_search_code_contents_by_name_success(tmp_path):
    """
    Test case for the `test_search_code_contents_by_name_success` function.
    
    This function tests that the `search_code_contents_by_name` method of the `JsonFileProcessor` class correctly searches for code and markdown content by name in a valid JSON file.
    
    Args:
        tmp_path: A temporary directory path.
    
    Returns:
        None.
    """
    payload = {"files": [{"name": "target", "code_content": "code", "md_content": ["md"]}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))

    processor = JsonFileProcessor(json_path)
    codes, mds = processor.search_code_contents_by_name(json_path, "target")

    assert codes == ["code"]
    assert mds == [["md"]]


def test_search_code_contents_by_name_handles_errors(tmp_path):
    """
    Test case for the `test_search_code_contents_by_name_handles_errors` function.
    
    This function tests that the `search_code_contents_by_name` method of the `JsonFileProcessor` class handles errors when the JSON file is invalid or missing.
    
    Args:
        tmp_path: A temporary directory path.
    
    Returns:
        None.
    """
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not-json")
    processor = JsonFileProcessor(invalid_path)

    assert processor.search_code_contents_by_name(invalid_path, "anything") == "Invalid JSON file."

    missing_path = tmp_path / "missing.json"
    assert processor.search_code_contents_by_name(missing_path, "anything") == "File not found."


def test_extract_data_handles_non_list_and_missing_content(tmp_path):
    """
    Test case for the `test_extract_data_handles_non_list_and_missing_content` function.
    
    This function tests that the `extract_data` method of the `JsonFileProcessor` class handles cases where the JSON data is missing or not in the expected list format.
    
    Args:
        tmp_path: A temporary directory path.
    
    Returns:
        None.
    """
    payload = {"file.py": {"unexpected": "structure"}, "other.py": [{"name": "noop"}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))

    processor = JsonFileProcessor(json_path)
    md_contents, extracted = processor.extract_data()

    assert md_contents == []
    assert extracted == []


def test_search_code_contents_by_name_no_match_and_exception(monkeypatch, tmp_path):
    """
    Test case for the `search_code_contents_by_name` method of the `JsonFileProcessor` class.
    
    This test case verifies the following scenarios:
    
    * When the `search_code_contents_by_name` method is called with a name that does not match any of the files in the JSON file, it should return a list of two strings: `["No matching item found."]` and `["No matching item found."]`.
    * When the `json.load` function raises an exception, the `search_code_contents_by_name` method should return the error message.
    
    Args:
        monkeypatch: A MonkeyPatch object to patch the `json.load` function.
        tmp_path: A temporary directory to write the JSON file to.
    
    Returns:
        None
    """
    payload = {"files": [{"name": "target", "code_content": "code", "md_content": ["md"]}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))
    processor = JsonFileProcessor(json_path)

    no_match_codes, no_match_mds = processor.search_code_contents_by_name(json_path, "missing")
    assert no_match_codes == ["No matching item found."]
    assert no_match_mds == ["No matching item found."]

    def boom(*_, **__):
        """
        A function that raises a `RuntimeError` with the message "boom".
        
        Args:
            *args: Variable-length argument list.
            **kwargs: Keyword arguments.
        
        Returns:
            None
        """
        raise RuntimeError("boom")

    monkeypatch.setattr(json, "load", boom)
    assert processor.search_code_contents_by_name(json_path, "target") == "An error occurred: boom"


def test_search_code_contents_by_name_missing_code_content(tmp_path):
    """
    Test case for the `search_code_contents_by_name` method of the `JsonFileProcessor` class.
    
    This test case verifies that the `search_code_contents_by_name` method returns the correct results when the code content for the given name is missing.
    
    Args:
        tmp_path: A temporary directory to write the JSON file to.
    
    Returns:
        None
    """
    payload = {"files": [{"name": "target", "md_content": ["md"]}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload))
    processor = JsonFileProcessor(json_path)

    codes, mds = processor.search_code_contents_by_name(json_path, "target")

    assert codes == ["No matching item found."]
    assert mds == ["No matching item found."]
