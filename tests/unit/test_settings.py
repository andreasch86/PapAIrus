import pytest
from pydantic import ValidationError

from papairus.settings import ChatCompletionSettings, LogLevel, ProjectSettings, SettingsManager


def test_language_validation_accepts_uk_english():
    settings = ProjectSettings(target_repo=".")
    assert settings.language == "English (UK)"


def test_language_validation_accepts_iso_code():
    settings = ProjectSettings(target_repo=".", language="en")
    assert settings.language == "English (UK)"


def test_language_validation_accepts_mixed_case_code():
    settings = ProjectSettings(target_repo=".", language="En")
    assert settings.language == "English (UK)"


def test_language_validation_rejects_other_languages():
    with pytest.raises(ValidationError):
        ProjectSettings(target_repo=".", language="French")


def test_language_validation_rejects_unknown_code():
    with pytest.raises(ValidationError):
        ProjectSettings(target_repo=".", language="zzzz")


def test_model_restriction():
    valid = ChatCompletionSettings(model="gemma-local")
    assert valid.model == "gemma-local"
    valid_gemini = ChatCompletionSettings(model="gemini-2.5-flash", gemini_api_key="key")
    assert valid_gemini.model == "gemini-2.5-flash"
    with pytest.raises(ValidationError):
        ChatCompletionSettings(model="gpt-4")


def test_gemini_key_required_only_for_gemini():
    with pytest.raises(ValidationError):
        ChatCompletionSettings(model="gemini-2.5-flash", gemini_api_key=None)

    settings = ChatCompletionSettings(model="gemma-local", gemini_api_key=None)
    assert settings.gemini_api_key is None


def test_log_level_validation_accepts_case_insensitive():
    settings = ProjectSettings(target_repo=".", log_level="debug")
    assert settings.log_level == LogLevel.DEBUG
    with pytest.raises(ValidationError):
        ProjectSettings(target_repo=".", log_level="trace")


def test_log_level_accepts_enum_value():
    settings = ProjectSettings(target_repo=".", log_level=LogLevel.WARNING)
    assert settings.log_level == LogLevel.WARNING


def test_log_level_validator_handles_enum_direct():
    assert ProjectSettings.set_log_level(LogLevel.ERROR) == LogLevel.ERROR


def test_log_level_validator_handles_non_string():
    with pytest.raises(ValueError):
        ProjectSettings.set_log_level(123)  # type: ignore[arg-type]


def test_base_url_casts_to_str():
    settings = ChatCompletionSettings(gemini_base_url="https://example.com", gemini_api_key="key")
    assert isinstance(settings.gemini_base_url, str)


def test_settings_manager_initialization(temp_repo):
    # Uses fixture to preconfigure settings
    setting = SettingsManager.get_setting()
    assert setting.project.target_repo.exists()
    assert setting.chat_completion.model == "gemini-2.5-flash"


def test_settings_manager_initialize_with_params_sets_instance(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    SettingsManager.initialize_with_params(
        target_repo=repo_path,
        markdown_docs_name="docs",
        hierarchy_name=".hier",
        ignore_list=[".venv"],
        language="English",
        max_thread_count=2,
        log_level="INFO",
        model="gemini-2.5-flash",
        temperature=0.3,
        request_timeout=30,
        gemini_base_url="https://example.com",
        telemetry_opt_in=True,
    )
    setting = SettingsManager.get_setting()
    assert setting.project.target_repo == repo_path
    assert setting.chat_completion.request_timeout == 30


def test_settings_module_as_script():
    import runpy

    runpy.run_module("papairus.settings", run_name="__main__")
