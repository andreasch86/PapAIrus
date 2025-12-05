import os
import sys
from importlib import import_module, metadata
from pathlib import Path

import click
import git
from pydantic import SecretStr, ValidationError

from papairus.doc_meta_info import DocItem, MetaInfo
from papairus.docstring_generator import DocstringGenerator
from papairus.exceptions import NoChangesWarning
from papairus.change_detector import ChangeDetector
from papairus.llm_provider import build_llm
from papairus.log import logger, set_logger_level_from_config
from papairus.runner import Runner
from papairus.settings import ChatCompletionSettings, LogLevel, SettingsManager
from papairus.utils.meta_info_utils import delete_fake_files, make_fake_files

try:
    version_number = metadata.version("papairus")
except metadata.PackageNotFoundError:  # pragma: no cover
    version_number = "0.0.0"


@click.group()
@click.version_option(version_number)
def cli():
    """An LLM-Powered Framework for Repository-level Code Documentation Generation."""
    pass


def handle_setting_error(e: ValidationError):
    """Handle configuration errors for settings."""
    for error in e.errors():
        field = error["loc"][-1]
        if error["type"] == "missing":
            message = click.style(
                f"Missing required field `{field}`. Please set the `{field}` environment variable.",
                fg="yellow",
            )
        else:
            message = click.style(error["msg"], fg="yellow")
        click.echo(message, err=True, color=True)

    raise click.ClickException(
        click.style("Program terminated due to configuration errors.", fg="red", bold=True)
    )


def _suggest_docs_refresh(repo: git.Repo, docs_path: Path, docs_folder_name: str) -> None:
    """Surface documentation refresh hints when docs already exist.

    Scans staged and unstaged Python changes and points to the docs
    directory so users can refresh narratives that reflect new work.
    """

    if not docs_path.exists():
        return

    change_detector = ChangeDetector(repo.working_tree_dir)
    staged = change_detector.get_staged_pys()
    unstaged = [
        diff.b_path for diff in repo.index.diff(None) if diff.b_path.endswith(".py")
    ]
    changed_files = set(staged.keys()) | set(unstaged)

    if not changed_files:
        click.echo(
            f"Docs directory {docs_path} already exists. No pending code changes detected."
        )
        return

    click.echo(
        f"Docs directory {docs_path} already exists. Consider refreshing docs for these changes:"
    )
    for file_path in sorted(changed_files):
        click.echo(f"- {file_path} -> review documentation under {docs_folder_name}/")


@cli.command(name="create-documentation")
@click.option(
    "--model",
    "-m",
    default="gemini-2.5-flash",
    show_default=True,
    help="Specifies the model to use for completion (gemma-local or any Gemini model starting with 'gemini-').",
    type=str,
)
@click.option(
    "--temperature",
    "-t",
    default=0.2,
    show_default=True,
    help="Sets the generation temperature for the model. Lower values make the model more deterministic.",
    type=float,
)
@click.option(
    "--request-timeout",
    "-r",
    default=60,
    show_default=True,
    help="Defines the timeout in seconds for the API request.",
    type=int,
)
@click.option(
    "--base-url",
    "-b",
    default="https://aiplatform.googleapis.com/v1",
    show_default=True,
    help="The base URL for Gemini API calls.",
    type=str,
)
@click.option(
    "--target-repo-path",
    "-tp",
    default="",
    show_default=True,
    help="The file system path to the target repository. This path is used as the root for documentation generation.",
    type=click.Path(file_okay=False),
)
@click.option(
    "--hierarchy-path",
    "-hp",
    default=".project_doc_record",
    show_default=True,
    help="The name or path for the project hierarchy file, used to organize documentation structure.",
    type=str,
)
@click.option(
    "--markdown-docs-path",
    "-mdp",
    default="docs",
    show_default=True,
    help="The folder path where Markdown documentation will be stored or generated.",
    type=str,
)
@click.option(
    "--ignore-list",
    "-i",
    default="",
    help="A comma-separated list of files or directories to ignore during documentation generation.",
)
@click.option(
    "--language",
    "-l",
    default="English (UK)",
    show_default=True,
    help="PapAIrus only supports UK English output.",
    type=str,
)
@click.option(
    "--max-thread-count",
    "-mtc",
    default=4,
    show_default=True,
)
@click.option(
    "--log-level",
    "-ll",
    default="INFO",
    show_default=True,
    help="Sets the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL) for the application. Default is INFO.",
    type=click.Choice([level.value for level in LogLevel], case_sensitive=False),
)
@click.option(
    "--print-hierarchy",
    "-pr",
    is_flag=True,
    show_default=True,
    default=False,
    help="If set, prints the hierarchy of the target repository when finished running the main task.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview actions and git diff without writing documentation.",
)
@click.option(
    "--allow-main",
    is_flag=True,
    default=False,
    help="Acknowledge running on the main branch (otherwise a warning is emitted).",
)
@click.option(
    "--telemetry/--no-telemetry",
    default=False,
    help="Explicitly opt in to telemetry (off by default).",
)
def create_documentation(
    model,
    temperature,
    request_timeout,
    base_url,
    target_repo_path,
    hierarchy_path,
    markdown_docs_path,
    ignore_list,
    language,
    max_thread_count,
    log_level,
    print_hierarchy,
    dry_run,
    allow_main,
    telemetry,
):
    """Generate project documentation leveraging code and existing docstrings."""
    repo = None
    try:
        repo = git.Repo(target_repo_path or ".")
    except git.InvalidGitRepositoryError:
        click.echo(
            click.style(
                "Target path is not a git repository; continuing without VCS safeguards.",
                fg="yellow",
            )
        )

    docs_path = Path(target_repo_path or ".") / markdown_docs_path

    if repo:
        _suggest_docs_refresh(repo, docs_path, markdown_docs_path)
        try:
            branch_name = repo.active_branch.name
            if branch_name in {"main", "master"} and not allow_main:
                click.echo(
                    click.style(
                        "Warning: running PapAIrus on the main branch. Re-run with --allow-main to proceed.",
                        fg="yellow",
                    ),
                    err=True,
                )
                return
        except TypeError:
            pass

        if repo and not repo.is_dirty(untracked_files=True):
            raise NoChangesWarning(
                "No code changes detected on this branch; PapAIrus will not run."
            )

    try:
        # Fetch and validate the settings using the SettingsManager
        _ = SettingsManager.initialize_with_params(
            target_repo=target_repo_path,
            hierarchy_name=hierarchy_path,
            markdown_docs_name=markdown_docs_path,
            ignore_list=[item.strip() for item in ignore_list.split(",") if item],
            language=language,
            log_level=log_level,
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            gemini_base_url=base_url,
            telemetry_opt_in=telemetry,
            max_thread_count=max_thread_count,
        )
        set_logger_level_from_config(log_level=log_level)
    except ValidationError as e:
        handle_setting_error(e)
        return

    if dry_run:
        click.echo("Dry-run: showing git diff and exiting without writes.")
        if repo:
            click.echo(repo.git.diff())
        else:
            click.echo("No git repository detected; nothing to diff.")
        return

    runner = Runner()
    runner.run()
    logger.success("Documentation task completed.")
    if print_hierarchy:
        runner.meta_info.target_repo_hierarchical_tree.print_recursive()
        logger.success("Hierarchy printed.")


@cli.command()
def clean():
    """Clean the fake files generated by the documentation process."""
    delete_fake_files()
    logger.success("Fake files have been cleaned up.")


@cli.command()
def diff():
    """Check for changes and print which documents will be updated or generated."""
    try:
        # Fetch and validate the settings using the SettingsManager
        setting = SettingsManager.get_setting()
    except ValidationError as e:
        handle_setting_error(e)
        return

    runner = Runner()
    if runner.meta_info.in_generation_process:  # English，English
        click.echo("This command only supports pre-check")
        raise click.Abort()

    file_path_reflections, jump_files = make_fake_files()
    new_meta_info = MetaInfo.init_meta_info(file_path_reflections, jump_files)
    new_meta_info.load_doc_from_older_meta(runner.meta_info)
    delete_fake_files()

    DocItem.check_has_task(
        new_meta_info.target_repo_hierarchical_tree,
        ignore_list=setting.project.ignore_list,
    )
    if new_meta_info.target_repo_hierarchical_tree.has_task:
        click.echo("The following docs will be generated/updated:")
        new_meta_info.target_repo_hierarchical_tree.print_recursive(
            diff_status=True, ignore_list=setting.project.ignore_list
        )
    else:
        click.echo("No docs will be generated/updated, check your source-code update")


@cli.command()
def chat_with_repo():
    """
    Start an interactive chat session with the repository.
    """
    try:
        # Fetch and validate the settings using the SettingsManager
        _ = SettingsManager.get_setting()
    except ValidationError as e:
        # Handle configuration errors if the settings are invalid
        handle_setting_error(e)
        return

    chat_module = sys.modules.get("papairus.chat_with_repo") or sys.modules.get(
        "papairus.chat_with_repo.main"
    )

    if chat_module is None:
        try:
            chat_module = import_module("papairus.chat_with_repo.main")
        except ModuleNotFoundError:
            chat_module = import_module("papairus.chat_with_repo")

    chat_main = getattr(chat_module, "main", None)
    if not callable(chat_main):  # pragma: no cover - defensive guard
        raise click.ClickException("papairus.chat_with_repo.main is not callable")

    chat_main()


@cli.command("generate-docstrings")
@click.option(
    "--path",
    "-p",
    default=".",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Root directory to scan for Python files.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview docstring updates without modifying files.",
)
@click.option(
    "--backend",
    type=click.Choice(["ast", "gemini", "gemma"], case_sensitive=False),
    default="ast",
    show_default=True,
    help="Docstring generation engine to use.",
)
@click.option(
    "--model",
    default=None,
    help=(
        "Model identifier to use for LLM-based docstrings. Defaults to a Gemini"
        " or Gemma model based on the backend."
    ),
)
@click.option(
    "--temperature",
    default=0.2,
    show_default=True,
    help="Sampling temperature for the LLM backend.",
)
@click.option(
    "--request-timeout",
    default=60,
    show_default=True,
    help="Request timeout (seconds) for LLM docstring generation.",
)
@click.option(
    "--gemini-base-url",
    default="https://aiplatform.googleapis.com/v1",
    show_default=True,
    help="Base URL for Gemini API calls.",
)
@click.option(
    "--gemini-api-key",
    default=None,
    envvar="GEMINI_API_KEY",
    show_envvar=True,
    help="API key for Gemini models.",
)
@click.option(
    "--ollama-base-url",
    default="http://localhost:11434",
    show_default=True,
    help="Base URL for Gemma (Ollama) models when using the Gemma backend.",
)
@click.option(
    "--ollama-model",
    default="gemma:2b",
    show_default=True,
    help="Ollama model name when using the Gemma backend.",
)
def generate_docstrings(
    path: Path,
    dry_run: bool,
    backend: str,
    model: str | None,
    temperature: float,
    request_timeout: int,
    gemini_base_url: str,
    gemini_api_key: str | None,
    ollama_base_url: str,
    ollama_model: str,
):
    """Add Google-style docstrings to callables missing complete documentation."""

    backend = backend.lower()
    llm_client = None

    if backend != "ast":
        selected_model = model
        if selected_model is None:
            selected_model = "gemini-2.5-flash" if backend == "gemini" else "gemma-local"

        resolved_gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        try:
            chat_settings = ChatCompletionSettings(
                model=selected_model,
                temperature=temperature,
                request_timeout=request_timeout,
                gemini_base_url=gemini_base_url,
                gemini_api_key=(
                    SecretStr(resolved_gemini_api_key)
                    if resolved_gemini_api_key
                    else None
                ),
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_model,
            )
            llm_client = build_llm(chat_settings)
        except Exception as exc:  # pragma: no cover - defensive guard for CLI output
            raise click.ClickException(str(exc))

    generator = DocstringGenerator(path, backend=backend, llm_client=llm_client)

    def _progress(path: Path, status: str) -> None:
        if status == "start":
            click.echo(f"[{backend}] Scanning {path}...", err=True)
        elif status == "updated":
            action = "Would update" if dry_run else "Updated"
            click.echo(f" -> {action} docstrings", err=True)
        elif status == "skipped":
            click.echo(" -> No docstring changes", err=True)

    updated_files = generator.run(dry_run=dry_run, progress_callback=_progress)

    if not updated_files:
        click.echo("No docstring updates needed.")
        return

    action = "Would update" if dry_run else "Updated"
    click.echo(f"{action} docstrings in {len(updated_files)} file(s):")
    for file_path in updated_files:
        click.echo(f"- {file_path}")


if __name__ == "__main__":  # pragma: no cover
    cli()
