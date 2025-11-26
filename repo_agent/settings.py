from enum import StrEnum
from pathlib import Path
from typing import Optional

from iso639 import Language, LanguageNotFoundError
from pydantic import (
    DirectoryPath,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ProjectSettings(BaseSettings):
    target_repo: DirectoryPath = ""  # type: ignore
    hierarchy_name: str = ".project_doc_record"
    markdown_docs_name: str = "markdown_docs"
    ignore_list: list[str] = []
    language: str = "English (UK)"
    max_thread_count: PositiveInt = 4
    log_level: LogLevel = LogLevel.INFO
    telemetry_opt_in: bool = False

    @field_validator("language")
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        normalized = v.strip().lower()
        if "english" in normalized:
            return "English (UK)"
        try:
            language_name = Language.match(v).name
        except LanguageNotFoundError:
            raise ValueError(
                "Invalid language input. Please enter a valid ISO 639 code or language name."
            )
        if language_name.lower() != "english":
            raise ValueError("PapAIrus only supports UK English output.")
        return "English (UK)"

    @field_validator("log_level", mode="before")
    @classmethod
    def set_log_level(cls, v: str) -> LogLevel:
        if isinstance(v, str):
            v = v.upper()  # Convert input to uppercase
        if v in LogLevel._value2member_map_:  # Check if the converted value is in enum members
            return LogLevel(v)
        raise ValueError(f"Invalid log level: {v}")


class ChatCompletionSettings(BaseSettings):
    model: str = "gemini-3.5-flash"  # Only Gemini (API) or local Gemma are allowed.
    temperature: PositiveFloat = 0.2
    request_timeout: PositiveInt = 60
    openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    openai_api_key: SecretStr = Field(..., exclude=True)

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def convert_base_url_to_str(cls, openai_base_url: HttpUrl) -> str:
        return str(openai_base_url)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        allowed_models = {"gemini-3.5-flash", "gemma-local"}
        if value not in allowed_models:
            raise ValueError(
                "Model must be one of: gemma-local (self-hosted) or gemini-3.5-flash (API)."
            )
        return value


class Setting(BaseSettings):
    project: ProjectSettings = {}  # type: ignore
    chat_completion: ChatCompletionSettings = {}  # type: ignore


class SettingsManager:
    _setting_instance: Optional[Setting] = None  # Private class attribute, initially None

    @classmethod
    def get_setting(cls):
        if cls._setting_instance is None:
            cls._setting_instance = Setting()
        return cls._setting_instance

    @classmethod
    def initialize_with_params(
        cls,
        target_repo: Path,
        markdown_docs_name: str,
        hierarchy_name: str,
        ignore_list: list[str],
        language: str,
        max_thread_count: int,
        log_level: str,
        model: str,
        temperature: float,
        request_timeout: int,
        openai_base_url: str,
        telemetry_opt_in: bool,
    ):
        project_settings = ProjectSettings(
            target_repo=target_repo,
            hierarchy_name=hierarchy_name,
            markdown_docs_name=markdown_docs_name,
            ignore_list=ignore_list,
            language=language,
            max_thread_count=max_thread_count,
            log_level=LogLevel(log_level),
            telemetry_opt_in=telemetry_opt_in,
        )

        chat_completion_settings = ChatCompletionSettings(
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            openai_base_url=openai_base_url,
        )

        cls._setting_instance = Setting(
            project=project_settings,
            chat_completion=chat_completion_settings,
        )


if __name__ == "__main__":
    setting = SettingsManager.get_setting()
    print(setting.model_dump())
