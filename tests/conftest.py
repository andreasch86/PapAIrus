import os
import sys
from pathlib import Path

import git
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from papairus.settings import (
    ChatCompletionSettings,
    LogLevel,
    ProjectSettings,
    Setting,
    SettingsManager,
)


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    SettingsManager._setting_instance = None
    yield
    SettingsManager._setting_instance = None


@pytest.fixture()
def temp_repo(tmp_path, monkeypatch):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo = git.Repo.init(repo_path)
    (repo_path / "sample.py").write_text("print('hi')\n")
    repo.index.add(["sample.py"])
    repo.index.commit("init")

    project_settings = ProjectSettings(
        target_repo=repo_path,
        hierarchy_name=".project_doc_record",
        markdown_docs_name="markdown_docs",
        ignore_list=[],
        language="English (UK)",
        max_thread_count=1,
        log_level=LogLevel.INFO,
        telemetry_opt_in=False,
    )
    chat_settings = ChatCompletionSettings()
    SettingsManager._setting_instance = Setting(
        project=project_settings, chat_completion=chat_settings
    )
    return repo
