import types

from papairus.chat_with_repo import main as chat_main_module
from papairus.settings import ChatCompletionSettings, ProjectSettings, Setting, SettingsManager


def test_select_repo_chat_settings_overrides_non_gemma():
    settings = ChatCompletionSettings(model="gemini-2.5-flash", gemini_api_key="key")

    overridden = chat_main_module._select_repo_chat_settings(settings)

    assert overridden.model == "codegemma"
    assert overridden.ollama_model == "codegemma:7b-instruct-q4_K_M"


def test_select_repo_chat_settings_keeps_gemma():
    settings = ChatCompletionSettings(model="codegemma")

    selected = chat_main_module._select_repo_chat_settings(settings)

    assert selected.model == "codegemma"
    assert selected.ollama_model == "codegemma:7b-instruct-q4_K_M"


def test_main_executes_with_stubs(monkeypatch, tmp_path):
    calls = {}

    def fake_get_setting():
        return Setting(
            project=ProjectSettings(target_repo=tmp_path),
            chat_completion=ChatCompletionSettings(model="codegemma"),
        )

    monkeypatch.setattr(SettingsManager, "get_setting", staticmethod(fake_get_setting))

    class DummyAssistant:
        def __init__(self, chat_settings, db_path):
            self.chat_settings = chat_settings
            self.db_path = db_path
            self.json_data = types.SimpleNamespace(
                extract_data=lambda: (["content"], [{"meta": True}])
            )
            self.vector_store_manager = types.SimpleNamespace(
                create_vector_store=lambda *_args: calls.setdefault("created", True)
            )

        def respond(self, *_args, **_kwargs):
            calls["respond"] = True

    monkeypatch.setattr(chat_main_module, "RepoAssistant", DummyAssistant)

    class DummyGradio:
        def __init__(self, callback):
            calls["gradio"] = callback

    monkeypatch.setattr(chat_main_module, "GradioInterface", DummyGradio)

    chat_main_module.main()

    assert calls["created"] is True
    assert "gradio" in calls
