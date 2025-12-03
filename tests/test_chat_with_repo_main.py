import types

import pytest

pytest.importorskip("gradio")

from papairus.chat_with_repo import main as chat_main
from papairus.settings import (
    ChatCompletionSettings,
    LogLevel,
    ProjectSettings,
    Setting,
)


def _stub_setting(tmp_path, model: str = "gemini-2.5-flash") -> Setting:
    project_settings = ProjectSettings(
        target_repo=tmp_path,
        hierarchy_name=".project_doc_record",
        markdown_docs_name="markdown_docs",
        ignore_list=[],
        language="English (UK)",
        max_thread_count=1,
        log_level=LogLevel.INFO,
        telemetry_opt_in=False,
    )
    chat_settings = ChatCompletionSettings(
        model=model,
        gemini_api_key="secret" if model.startswith("gemini-") else None,
    )
    return Setting(project=project_settings, chat_completion=chat_settings)


def test_select_repo_chat_settings_overrides_gemini():
    settings = ChatCompletionSettings(model="gemini-2.5-flash", gemini_api_key="secret")
    updated = chat_main._select_repo_chat_settings(settings)

    assert updated.model == "gemma-local"
    assert updated.temperature == settings.temperature


def test_select_repo_chat_settings_keeps_gemma():
    settings = ChatCompletionSettings(model="gemma-local")

    assert chat_main._select_repo_chat_settings(settings) is settings


def test_chat_with_repo_uses_gemma_only(monkeypatch, tmp_path):
    captured = {}

    def fake_get_setting():
        return _stub_setting(tmp_path, model="gemini-2.5-flash")

    monkeypatch.setattr(chat_main.SettingsManager, "get_setting", fake_get_setting)

    class DummyAssistant:
        def __init__(self, chat_settings, db_path):
            captured["model"] = chat_settings.model
            captured["db_path"] = db_path
            self.json_data = types.SimpleNamespace(extract_data=lambda: ([], []))
            self.vector_store_manager = types.SimpleNamespace(
                create_vector_store=lambda *_: captured.setdefault("store", True)
            )
            self.respond = lambda *_: None

    monkeypatch.setattr(chat_main, "RepoAssistant", DummyAssistant)
    monkeypatch.setattr(chat_main, "GradioInterface", lambda *_: captured.setdefault("ui", True))

    chat_main.main()

    assert captured["model"] == "gemma-local"
    assert captured["store"] is True
    assert captured["ui"] is True
