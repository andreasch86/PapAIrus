# PapAIrus (papairus)

PapAIrus is a corporate adaptation of the upstream [OpenBMB/RepoAgent](https://github.com/OpenBMB/RepoAgent), maintained by the AI Platform team and tailored by Andrea Scholtz for internal documentation workflows. The project enforces UK English output, Google-style docstrings, and a constrained model set (local Gemma or Google Gemini 3.5, Flash when available) to keep behaviour predictable and reviewable.

## Key capabilities
- Automatic repository analysis and documentation generation across Python, Go, Rust, C++, Java, and SQL projects.
- Safety rails: refuses to run when no code changes are present, warns before operating on `main`/`master`, and provides a `--dry-run` preview.
- Telemetry is opt-in only; no background tracking unless explicitly requested.
- Config discovery prioritises `pyproject.toml` `[tool.corp_repoagent]` in the target repository, with CLI flags as a fallback.
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
PapAIrus exposes the `papairusrun` entry point.

```bash
export OPENAI_API_KEY=your_gemini_key
papairusrun run --model gemini-3.5-flash --allow-main --telemetry
papairusrun run --model gemma-local --dry-run
```

Options:
- `--allow-main`: acknowledge running on protected branches.
- `--dry-run`: show planned actions and git diff without writing files.
- `--telemetry/--no-telemetry`: explicit opt-in/out switch (default off).
- `--language` defaults to `English (UK)`; other inputs are rejected.

Cleaning and diff helpers:
```bash
papairusrun clean
papairusrun diff
```

## Development notes
- Pre-commit hooks should install the agent from the private GitLab mirror rather than local sources.
- When a `docs/` directory exists in a target project, link its Markdown files from that project's README to keep navigation centralised.
- HTML book outputs live alongside generated docs; link them from the target README so users can find rendered content easily.

## Licensing
PapAIrus remains under the Apache 2.0 license while incorporating corporate safeguards and defaults.
