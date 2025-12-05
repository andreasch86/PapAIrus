import runpy
import sys
import types
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from papairus import main
from papairus.settings import (
    ChatCompletionSettings,
    LogLevel,
    ProjectSettings,
    Setting,
    SettingsManager,
)


def stub_settings(repo_path):
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
    chat_settings = ChatCompletionSettings(gemini_api_key="secret")
    return Setting(project=project_settings, chat_completion=chat_settings)


def test_run_blocks_clean_repo(temp_repo):
    repo_path = temp_repo.working_tree_dir
    runner = CliRunner()
    result = runner.invoke(
        main.run,
        ["--target-repo-path", repo_path, "--allow-main"],
    )
    assert result.exit_code != 0
    assert "No code changes detected" in result.output


def test_cli_group_callable():
    main.cli.callback()


def test_run_dry_run_outputs_diff(temp_repo, monkeypatch):
    repo_path = temp_repo.working_dir
    repo_path_obj = Path(temp_repo.working_tree_dir)
    (repo_path_obj / "sample.py").write_text("print('changed')\n")

    monkeypatch.setattr(
        SettingsManager,
        "initialize_with_params",
        lambda **kwargs: stub_settings(repo_path_obj),
    )

    runner = CliRunner()
    result = runner.invoke(
        main.run,
        [
            "--target-repo-path",
            repo_path,
            "--allow-main",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry-run" in result.output
    assert "sample.py" in result.output


def test_run_warns_on_main_branch(monkeypatch):
    class FakeBranch:
        name = "main"

    class FakeRepo:
        active_branch = FakeBranch()

        def is_dirty(self, untracked_files=True):
            return True

    monkeypatch.setattr(main.git, "Repo", lambda _: FakeRepo())

    runner = CliRunner()
    result = runner.invoke(main.run, ["--target-repo-path", "."])
    assert result.exit_code == 0
    assert "Warning: running PapAIrus on the main branch" in result.output


def test_run_handles_detached_head(monkeypatch, tmp_path):
    class FakeRepo:
        @property
        def active_branch(self):
            raise TypeError("detached")

        def is_dirty(self, untracked_files=True):
            return True

    monkeypatch.setattr(main.git, "Repo", lambda _: FakeRepo())
    monkeypatch.setattr(main, "Runner", lambda: types.SimpleNamespace(run=lambda: None, meta_info=types.SimpleNamespace(target_repo_hierarchical_tree=None)))
    monkeypatch.setattr(
        SettingsManager,
        "initialize_with_params",
        lambda **kwargs: stub_settings(tmp_path),
    )

    runner = CliRunner()
    result = runner.invoke(main.run, ["--target-repo-path", str(tmp_path), "--allow-main"])
    assert result.exit_code == 0


def test_run_handles_invalid_repo_and_runs(monkeypatch, tmp_path):
    repo_path = tmp_path / "missing"
    repo_path.mkdir()
    monkeypatch.setattr(
        main.git,
        "Repo",
        lambda _: (_ for _ in ()).throw(main.git.InvalidGitRepositoryError()),
    )

    captured = {}

    class DummyRunner:
        def __init__(self):
            self.meta_info = types.SimpleNamespace(target_repo_hierarchical_tree=None)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(main, "Runner", DummyRunner)
    monkeypatch.setattr(
        SettingsManager,
        "initialize_with_params",
        lambda **kwargs: stub_settings(repo_path),
    )

    runner = CliRunner()
    result = runner.invoke(
        main.run,
        ["--target-repo-path", str(repo_path), "--allow-main", "--model", "gemini-2.5-flash"],
    )

    assert result.exit_code == 0
    assert captured["ran"] is True
    assert "not a git repository" in result.output


def test_run_handles_validation_error(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    class SampleModel(click.ParamType):
        name = "sample"

    from pydantic import BaseModel, ValidationError

    class Foo(BaseModel):
        val: int

    try:
        Foo(val="bad")
    except ValidationError as err:
        validation_error = err

    monkeypatch.setattr(main.git, "Repo", lambda _: types.SimpleNamespace(active_branch=types.SimpleNamespace(name="feature"), is_dirty=lambda untracked_files=True: True))

    called = {}
    monkeypatch.setattr(main, "handle_setting_error", lambda e: called.setdefault("handled", True))
    monkeypatch.setattr(main.SettingsManager, "initialize_with_params", lambda **kwargs: (_ for _ in ()).throw(validation_error))

    runner = CliRunner()
    result = runner.invoke(main.run, ["--target-repo-path", str(repo_path), "--allow-main"])
    assert result.exit_code == 0
    assert called["handled"] is True


def test_run_dry_run_without_repo(monkeypatch, tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    monkeypatch.setattr(main.git, "Repo", lambda _: (_ for _ in ()).throw(main.git.InvalidGitRepositoryError()))
    monkeypatch.setattr(main.SettingsManager, "initialize_with_params", lambda **kwargs: stub_settings(repo_path))

    runner = CliRunner()
    result = runner.invoke(main.run, ["--target-repo-path", str(repo_path), "--allow-main", "--dry-run"])
    assert result.exit_code == 0
    assert "nothing to diff" in result.output


def test_run_executes_happy_path(monkeypatch, temp_repo):
    repo_path = temp_repo.working_tree_dir
    (Path(repo_path) / "new_file.py").write_text("print('new')\n")

    class DummyRunner:
        def __init__(self):
            self.meta_info = types.SimpleNamespace(target_repo_hierarchical_tree=None)
            self.ran = False

        def run(self):
            self.ran = True

    runner_instance = DummyRunner()
    monkeypatch.setattr(main, "Runner", lambda: runner_instance)
    monkeypatch.setattr(
        SettingsManager,
        "initialize_with_params",
        lambda **kwargs: stub_settings(Path(repo_path)),
    )

    cli_runner = CliRunner()
    result = cli_runner.invoke(
        main.run,
        ["--target-repo-path", repo_path, "--allow-main", "--log-level", "INFO"],
    )

    assert result.exit_code == 0
    assert runner_instance.ran is True


def test_run_prints_hierarchy(monkeypatch, temp_repo):
    repo_path = Path(temp_repo.working_tree_dir)
    (repo_path / "another.py").write_text("print('touch')\n")

    printed = {}

    class DummyRunner:
        def __init__(self):
            self.meta_info = types.SimpleNamespace(
                target_repo_hierarchical_tree=types.SimpleNamespace(
                    print_recursive=lambda: printed.setdefault("printed", True)
                )
            )
            self.ran = False

        def run(self):
            self.ran = True

    runner_instance = DummyRunner()
    monkeypatch.setattr(main, "Runner", lambda: runner_instance)
    monkeypatch.setattr(main.SettingsManager, "initialize_with_params", lambda **kwargs: stub_settings(repo_path))

    cli_runner = CliRunner()
    result = cli_runner.invoke(
        main.run,
        [
            "--target-repo-path",
            str(repo_path),
            "--allow-main",
            "--print-hierarchy",
        ],
    )
    assert result.exit_code == 0
    assert printed["printed"] is True


def test_clean_invokes_delete(monkeypatch):
    invoked = {}
    monkeypatch.setattr(main, "delete_fake_files", lambda: invoked.setdefault("clean", True))
    runner = CliRunner()
    result = runner.invoke(main.clean)
    assert result.exit_code == 0
    assert invoked["clean"] is True


def test_diff_aborts_when_mid_generation(monkeypatch):
    class FakeMeta:
        in_generation_process = True

    class FakeRunner:
        meta_info = FakeMeta()

    monkeypatch.setattr(main, "Runner", lambda: FakeRunner())
    runner = CliRunner()
    result = runner.invoke(main.diff)
    assert result.exit_code != 0
    assert "pre-check" in result.output


def test_diff_handles_validation_error(monkeypatch):
    from pydantic import BaseModel, ValidationError

    class Foo(BaseModel):
        value: int

    try:
        Foo(value="bad")
    except ValidationError as err:
        validation_error = err

    captured = {}
    monkeypatch.setattr(main.SettingsManager, "get_setting", lambda: (_ for _ in ()).throw(validation_error))
    monkeypatch.setattr(main, "handle_setting_error", lambda e: captured.setdefault("handled", True))

    runner = CliRunner()
    result = runner.invoke(main.diff)
    assert result.exit_code == 0
    assert captured["handled"] is True


def test_diff_reports_tasks(monkeypatch):
    fake_tree = types.SimpleNamespace(
        has_task=True,
        print_recursive=lambda diff_status, ignore_list: None,
    )

    class FakeMeta:
        in_generation_process = False

        def __init__(self):
            self.target_repo_hierarchical_tree = fake_tree

        def load_doc_from_older_meta(self, *_):
            pass

    class FakeMetaFactory:
        @classmethod
        def init_meta_info(cls, reflections, jumps):
            return FakeMeta()

    class FakeRunner:
        def __init__(self):
            self.meta_info = FakeMeta()

    monkeypatch.setattr(main, "Runner", FakeRunner)
    monkeypatch.setattr(main, "MetaInfo", FakeMetaFactory)
    monkeypatch.setattr(main, "make_fake_files", lambda: ([], []))
    monkeypatch.setattr(main, "delete_fake_files", lambda: None)
    monkeypatch.setattr(main, "DocItem", types.SimpleNamespace(check_has_task=lambda *args, **kwargs: None))
    monkeypatch.setattr(
        SettingsManager,
        "get_setting",
        lambda: types.SimpleNamespace(project=types.SimpleNamespace(ignore_list=[], target_repo=Path("."))),
    )

    runner = CliRunner()
    result = runner.invoke(main.diff)
    assert result.exit_code == 0
    assert "generated/updated" in result.output


def test_diff_reports_no_tasks(monkeypatch):
    fake_tree = types.SimpleNamespace(
        has_task=False,
        print_recursive=lambda diff_status, ignore_list: None,
    )

    class FakeMeta:
        in_generation_process = False

        def __init__(self):
            self.target_repo_hierarchical_tree = fake_tree

        def load_doc_from_older_meta(self, *_):
            pass

    class FakeMetaFactory:
        @classmethod
        def init_meta_info(cls, reflections, jumps):
            return FakeMeta()

    class FakeRunner:
        def __init__(self):
            self.meta_info = FakeMeta()

    monkeypatch.setattr(main, "Runner", FakeRunner)
    monkeypatch.setattr(main, "MetaInfo", FakeMetaFactory)
    monkeypatch.setattr(main, "make_fake_files", lambda: ([], []))
    monkeypatch.setattr(main, "delete_fake_files", lambda: None)
    monkeypatch.setattr(main, "DocItem", types.SimpleNamespace(check_has_task=lambda *args, **kwargs: None))
    monkeypatch.setattr(
        SettingsManager,
        "get_setting",
        lambda: types.SimpleNamespace(project=types.SimpleNamespace(ignore_list=[], target_repo=Path("."))),
    )

    runner = CliRunner()
    result = runner.invoke(main.diff)
    assert result.exit_code == 0
    assert "No docs will be generated" in result.output


def test_chat_with_repo_invokes_module(monkeypatch):
    invoked = {}
    fake_chat_module = types.SimpleNamespace(main=lambda: invoked.setdefault("chat", True))
    monkeypatch.setitem(sys.modules, "papairus.chat_with_repo", fake_chat_module)
    monkeypatch.setattr(SettingsManager, "get_setting", lambda: stub_settings(Path(".")))

    runner = CliRunner()
    result = runner.invoke(main.chat_with_repo)
    assert result.exit_code == 0
    assert invoked["chat"] is True


def test_chat_with_repo_handles_validation_error(monkeypatch):
    from pydantic import BaseModel, ValidationError

    class Foo(BaseModel):
        value: int

    try:
        Foo(value="bad")
    except ValidationError as err:
        validation_error = err

    handled = {}
    monkeypatch.setattr(main.SettingsManager, "get_setting", lambda: (_ for _ in ()).throw(validation_error))
    monkeypatch.setattr(main, "handle_setting_error", lambda e: handled.setdefault("handled", True))

    runner = CliRunner()
    result = runner.invoke(main.chat_with_repo)
    assert result.exit_code == 0
    assert handled["handled"] is True


def test_handle_setting_error_outputs_messages(capsys):
    error = types.SimpleNamespace(
        errors=lambda: [
            {"loc": ["project", "gemini_api_key"], "type": "missing", "msg": "field required"},
            {"loc": ["project", "language"], "type": "value_error", "msg": "bad language"},
        ]
    )

    with pytest.raises(click.ClickException):
        main.handle_setting_error(error)

    captured = capsys.readouterr()
    assert "Missing required field `gemini_api_key`" in captured.err
    assert "bad language" in captured.err


def test_module_entrypoint_executes(monkeypatch):
    calls = {}

    def fake_cli():
        calls["cli"] = True

    monkeypatch.setattr(main, "cli", fake_cli)
    runpy.run_module("papairus.__main__", run_name="__main__")
    assert calls["cli"] is True


def test_module_entrypoint_import_does_not_trigger_cli():
    import importlib

    calls = {}

    def fake_cli():
        calls["cli"] = True

    import papairus.__main__ as module

    module.cli = fake_cli
    importlib.reload(module)
    assert calls == {}


def test_generate_docstrings_cli_dry_run(tmp_path):
    sample = tmp_path / "cli_sample.py"
    sample.write_text("def hello(name):\n    return name\n")

    runner = CliRunner()
    result = runner.invoke(
        main.generate_docstrings, ["--path", str(tmp_path), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "Would update docstrings" in result.output
    assert sample.read_text().count('"""') == 0


def test_generate_docstrings_cli_with_llm(monkeypatch, tmp_path):
    created = {}
    captured_settings = {}

    class DummyGenerator:
        def __init__(self, root, backend="ast", llm_client=None, **_):
            created["backend"] = backend
            created["llm_client"] = llm_client

        def run(self, dry_run=False):
            created["dry_run"] = dry_run
            return [tmp_path / "file.py"]

    monkeypatch.setattr(main, "DocstringGenerator", DummyGenerator)
    def record_settings(settings):
        captured_settings["api_key"] = settings.gemini_api_key.get_secret_value()
        return "llm"

    monkeypatch.setattr(main, "build_llm", record_settings)
    monkeypatch.setenv("GEMINI_API_KEY", "env-secret")

    runner = CliRunner()
    result = runner.invoke(
        main.generate_docstrings,
        [
            "--path",
            str(tmp_path),
            "--backend",
            "gemini",
        ],
    )

    assert result.exit_code == 0
    assert created["backend"] == "gemini"
    assert created["llm_client"] == "llm"
    assert captured_settings["api_key"] == "env-secret"
    assert "docstrings in 1 file" in result.output


def test_chat_with_repo_imports_fallback(monkeypatch):
    dummy_module = types.SimpleNamespace(main=lambda: None)
    attempts = {"count": 0}

    def fake_import_module(name):
        attempts["count"] += 1
        if name == "papairus.chat_with_repo.main" and attempts["count"] == 1:
            raise ModuleNotFoundError
        assert name in {"papairus.chat_with_repo.main", "papairus.chat_with_repo"}
        return dummy_module

    for key in ["papairus.chat_with_repo", "papairus.chat_with_repo.main"]:
        sys.modules.pop(key, None)

    monkeypatch.setattr(main, "import_module", fake_import_module)

    runner = CliRunner()
    result = runner.invoke(main.chat_with_repo)

    assert result.exit_code == 0
    assert attempts["count"] == 2
