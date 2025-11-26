# PapAIrus

PapAIrus (`papairus`) is a corporate adaptation of the upstream [OpenBMB/RepoAgent](https://github.com/OpenBMB/RepoAgent), maintained by the AI Platform team and tailored by Andrea Scholtz for internal documentation workflows. The project enforces UK English output, Google-style docstrings, and a constrained model set (local Gemma or Google Gemini 3.5, Flash when available) to keep behaviour predictable and reviewable.

## Key capabilities
- Automatic repository analysis and documentation generation across Python, Go, Rust, C++, Java, and SQL projects.
- Safety rails: refuses to run when no code changes are present, warns before operating on `main`/`master`, and provides a `--dry-run` preview.
- Telemetry is opt-in only; no background tracking unless explicitly requested.
- Config discovery prioritises `pyproject.toml` `[tool.corp_papairus]` in the target repository, with CLI flags as a fallback.
- GitLab integration (planned): generate docs, push to a branch, and open an MR with reviewers notified.

## Installation
PapAIrus uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/andreasch86/PapAIrus.git
cd PapAIrus
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[chat_with_repo]
```

## Usage
PapAIrus exposes the `papairus` command for documentation generation and an interactive chat UI for RAG-style question answering.

### Model configuration (Gemini vs. Gemma/Ollama)
- **Gemini API**: set `GEMINI_API_KEY` and optionally override `--base-url`.
- **Gemma via Ollama**: set `--model gemma-local` and point to your Ollama deployment with `OLLAMA_BASE_URL`/`--base-url` (defaults to `http://localhost:11434`). Gemini keys are not required for local Gemma.
- Embeddings follow the same provider as the model: Gemini embeddings for cloud, or `ollama_embedding_model` for local usage.

### Generate repository documentation (CLI)
Run from the target repository root (or pass `--target-repo-path`):

```bash
# Gemini cloud generation
export GEMINI_API_KEY=your_gemini_key
papairus run --model gemini-3.5-flash --allow-main --telemetry

# Local Gemma via Ollama
papairus run --model gemma-local --base-url http://localhost:11434 --dry-run
```

Key flags:
- `--allow-main`: acknowledge running on protected branches.
- `--dry-run`: show planned actions and git diff without writing files.
- `--telemetry/--no-telemetry`: explicit opt-in/out switch (default off).
- `--language` defaults to `English (UK)`; other inputs are rejected.
- `--markdown-docs-path` and `--hierarchy-path` control where generated Markdown files and hierarchy metadata are written (default `markdown_docs` and `.project_doc_record`).

Housekeeping helpers:
```bash
papairus clean
papairus diff
```

### Interactive chat over generated docs
The chat UI uses the existing hierarchy (`.project_doc_record/project_hierarchy.json`) and Markdown outputs. After you have run `papairus run` at least once on the repository:

```bash
# Launches a Gradio interface at http://localhost:7860/ by default
papairus chat-with-repo
```

You can ask free-form questions about the repository, with responses grounded in the generated embeddings and code snippets. Model selection follows the same Gemini/Gemma rules as the CLI.

### Render GitBooks for the generated Markdown
Use the `display/` helpers to build and serve a GitBook from the Markdown docs:

1. Create `config.yml` at the repository root to point the GitBook tools to the right paths:
   ```yaml
   repo_path: /absolute/path/to/your/repo
   Markdown_Docs_folder: markdown_docs
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
- Pre-commit hooks should install the agent from the private GitLab mirror rather than local sources.
- When a `docs/` directory exists in a target project, link its Markdown files from that project's README to keep navigation centralised.
- HTML book outputs live alongside generated docs; link them from the target README so users can find rendered content easily.

## Licensing
PapAIrus remains under the Apache 2.0 license while incorporating corporate safeguards and defaults.
