# PapAIrus

PapAIrus (`papairus`) is a documentation-focused fork of [OpenBMB/RepoAgent](https://github.com/OpenBMB/RepoAgent) designed for predictable, reviewable outputs (UK English only) and Google-style docstrings. It supports modular LLM backends for both chat and docstring generation, with first-class Gemini and Ollama-powered CodeGemma/Gemma models.

## Key capabilities
- Automatic repository analysis and documentation generation across Python, Go, Rust, C++, Java, and SQL projects.
- Safety rails: refuses to run when no code changes are present, warns before operating on `main`/`master`, and provides a `--dry-run` preview.
- Modular LLM engine with interchangeable providers (Gemini or local Gemma/CodeGemma via Ollama) and distinct chat vs. docstring prompt flows.
- Telemetry is opt-in only; no background tracking unless explicitly requested.
- Config discovery prioritises `pyproject.toml` `[tool.corp_papairus]` in the target repository, with CLI flags as a fallback.

## Prerequisites
- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- Optional: [Ollama](https://ollama.com) for local Gemma/CodeGemma models

## Installation
```bash
git clone https://github.com/andreasch86/PapAIrus.git
cd PapAIrus
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[chat_with_repo]
```

### Installing Ollama
PapAIrus can auto-pull Ollama models when missing, but Ollama itself must be installed and running.

- **macOS** (Homebrew):
  ```bash
  brew install ollama
  ollama serve  # start the daemon
  ```
- **Linux** (official install script):
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama serve
  ```
- **Windows** (PowerShell):
  ```powershell
  winget install Ollama.Ollama
  # After installation completes, start the Ollama service
  ollama serve
  ```

By default, PapAIrus targets the `codegemma:7b-instruct-q4_K_M` model. If it is not present, the backend triggers `POST /api/pull` automatically before the first request. Auto-pull can be disabled via configuration if you prefer manual model management.

## Backend configuration
PapAIrus exposes a unified interface for chat and docstring generation. Choose a backend per command:
- `ast`: offline docstring extraction (no network calls)
- `gemini`: Google Gemini models (requires `GEMINI_API_KEY`)
- `gemma`: Local Gemma/CodeGemma via Ollama. The Ollama model tag is configured via `--ollama-model` (defaults to `codegemma:7b-instruct-q4_K_M`).

Defaults:
- Local model selector: `codegemma` (uses Ollama `codegemma:7b-instruct-q4_K_M` by default)
- Ollama base URL: `http://localhost:11434`
- Ollama model tag: `codegemma:7b-instruct-q4_K_M` (auto-pulled if missing)
- Chat/documentation pipeline model selector: `codegemma` chooses the Ollama-backed engine, while any `gemini-*` value targets Gemini.

Set the Gemini API key through `GEMINI_API_KEY` or `--gemini-api-key`. Ollama endpoints require no API key.

## Usage
PapAIrus provides a CLI for generating docstrings and repository documentation plus an interactive chat UI for RAG-style Q&A.

### Generate or preview docstrings for Python files
```bash
# AST-only generation (no network calls)
papairus generate-docstrings --path path/to/project --dry-run

# CodeGemma via Ollama (auto-pulls codegemma:7b-instruct-q4_K_M if missing)
papairus generate-docstrings --path path/to/project --backend gemma \
  --ollama-base-url http://localhost:11434 --ollama-model codegemma:7b-instruct-q4_K_M

# Gemini-backed generation (optional; requires an API key)
export GEMINI_API_KEY=your_gemini_key
papairus generate-docstrings --path path/to/project --backend gemini \
  --model gemini-2.5-flash --temperature 0.2
```
Key options:
- `--backend`: choose `ast` (default), `gemini`, or `gemma`.
- `--dry-run`: show which files would change without writing them.
- `--model`: override the default model per backend.
- `--gemini-base-url`/`--gemini-api-key`: control Gemini endpoints and auth.
- `--ollama-base-url`/`--ollama-model`: control the Gemma/CodeGemma Ollama endpoint and model. Auto-pull is enabled by default and can be disabled via configuration if needed.

### Generate repository documentation (CLI)
Run from the target repository root (or pass `--target-repo-path`):

```bash
# Local CodeGemma via Ollama (default)
papairus create-documentation --model codegemma --base-url http://localhost:11434 --dry-run

# Gemini cloud generation (optional)
export GEMINI_API_KEY=your_gemini_key
papairus create-documentation --model gemini-2.5-flash --allow-main --telemetry
```
Key flags:
- `--allow-main`: acknowledge running on protected branches.
- `--dry-run`: show planned actions and git diff without writing files.
- `--telemetry/--no-telemetry`: explicit opt-in/out switch (default off).
- `--language` defaults to `English (UK)`; other inputs are rejected.
- `--markdown-docs-path` and `--hierarchy-path` control where generated Markdown files and hierarchy metadata are written (default `docs` and `.project_doc_record`).

### Interactive chat over generated docs
After running `papairus create-documentation` at least once:

```bash
# Launches a Gradio interface at http://localhost:7860/ by default
papairus chat-with-repo
```
`chat-with-repo` always uses the local CodeGemma instruct model served by Ollama (auto-pulled if missing). The chat pipeline injects repository context before sending prompts, so ensure the Ollama service is running and has network access to download `codegemma:7b-instruct-q4_K_M` on first launch.

### Render GitBooks for the generated Markdown
Use the `display/` helpers to build and serve a GitBook from the Markdown docs:

1. Create `config.yml` at the repository root to point the GitBook tools to the right paths:
   ```yaml
   repo_path: /absolute/path/to/your/repo
   Markdown_Docs_folder: docs
   ```
2. Install Node.js 10.x and initialise GitBook (one-time):
   ```bash
   cd display
   make init_env   # installs Node.js 10 via nvm
   make init       # installs gitbook-cli and plugins
   ```
3. Generate and serve the book whenever docs change:
   ```bash
   make serve
   ```
   This builds the book into `display/books/<repo_name>` and serves it at http://localhost:4000/.

The `display/README_DISPLAY.md` file contains more operational details if you need to customise the GitBook pipeline.

## Development notes
- Pre-commit hooks must be run locally: `uv run pre-commit run --all-files`.
- When a `docs/` directory exists in a target project, link its Markdown files from that project's README to keep navigation centralised.
- HTML book outputs live alongside generated docs; link them from the target README so users can find rendered content easily.

## Licensing
PapAIrus remains under the Apache 2.0 license while incorporating corporate safeguards and defaults.
