# LLM Engine Refactor: Interface and Backend Split

## Current Observations

- `papairus/llm_provider.py` builds either a Gemini-based chat client (`VertexGeminiLLM`) or a local Gemma client through llama-index Ollama bindings, keyed off `ChatCompletionSettings.model`. The logic is centralized and tightly coupled to llama-index primitives.
- `papairus/docstring_generator.py` relies on a callable `llm_client` that exposes a `chat(messages)` method and constructs prompts inline when the backend is not the built-in AST mode.
- `papairus/chat_engine.py` pulls an LLM via `build_llm` and formats prompts using `chat_template`, with no explicit distinction between conversational chat and docstring generation modes.

## Proposed File Changes (Step 1: LLM Interface Abstraction)

- **`papairus/llm_provider.py`**
  - Replace the direct factory methods with an `LLMBackend` abstract base defining core chat and docstring-generation entry points (e.g., `generate_response` and `generate_docstring`).
  - Move provider-specific initialization details out of this module; convert it into a registry/loader that maps configuration values to backend classes.
- **`papairus/llm/backends/base.py`** (new)
  - Define the `LLMBackend` ABC with chat and docstring methods and common dataclass-style configuration objects for temperature, timeouts, and model identifiers.
  - Include lightweight response objects (message content plus token usage metadata) that mirror the shapes currently expected by `ChatEngine` and `DocstringGenerator`.
- **`papairus/llm/backends/gemini.py`** (new)
  - Implement `GeminiBackend` using the existing `VertexGeminiLLM` logic but adapted to the new base interface.
  - Keep request building/response parsing isolated so tests can mock network calls without pulling in external dependencies.
- **`papairus/llm/backends/local_gemma.py`** (new)
  - Implement `LocalGemmaBackend` targeting `llama-cpp-python` (or `langchain_community.llms.LlamaCpp`) with configurable model paths for CodeGemma 2B, 7B, and 7B Q4_K_M variants.
  - Encapsulate model-loading options (context length, GPU/CPU flags) while conforming to the shared `LLMBackend` interface.
  - Add optional Ollama integration that checks for the configured model locally and automatically triggers an `ollama pull <model>` when missing. Provide an opt-out flag and emit clear logging around model discovery/pull events so CI can mock the behavior without network calls.
- **`papairus/settings.py`**
  - Add explicit engine selection/configuration fields (e.g., `llm_engine`, `model_path`, `api_key`) and surface environment/config file wiring to switch backends without code changes.
- **`tests/` (new/updated)**
  - Add unit tests for the `LLMBackend` contract (ensuring required methods and error paths) using mocks so no real models load during CI.
  - Cover backend selection/registry behavior when different engines are configured.

## Proposed File Changes (Step 2: Chat vs. Docstring Engine Split)

- **`papairus/docstring_generator.py`**
  - Route LLM calls through the `LLMBackend.generate_docstring` method, passing the AST-isolated code snippet and style metadata instead of raw prompt strings.
  - Move few-shot, Google-style docstring prompting into the backend layer to keep generator logic focused on AST extraction and indentation.
- **`papairus/chat_engine.py`**
  - Use `LLMBackend.generate_response` for conversational mode and support injection of repository context blocks (file summaries, directory structure) before relaying the user prompt.
  - Maintain compatibility with the existing `chat_template` helper by adapting it to assemble the context-aware prompt payloads for the new interface.
- **`papairus/prompt.py`**
  - Split prompt utilities into chat mode (conversational with context block) and docstring mode (strict few-shot, Google-style output with no filler).
- **`papairus/settings.py`**
  - Distinguish configuration for chat vs. docstring modes (e.g., separate temperatures or model names if needed) while mapping both to the unified `LLMBackend` loader.
- **`tests/` (new/updated)**
  - Mock `LLMBackend` to validate that `DocstringGenerator` injects AST snippets and style parameters into `generate_docstring`.
  - Add chat-mode tests ensuring repository context is included before delegating to the backend.

## README Overhaul and Local Setup Notes

- Replace the current `README.md` with a fresh end-to-end guide covering project overview, prerequisites, installation via `uv`, and the new modular LLM configuration model.
- Add a dedicated section for installing Ollama (macOS/Linux) and verifying the daemon is running, plus a short note on Windows WSL usage if relevant.
- Document engine selection through `config.yaml`/`.env` (e.g., `LLM_ENGINE=gemini-pro`, `LLM_ENGINE=codegemma-7b-quantized`) and show CLI examples for chat vs. docstring modes.
- Include troubleshooting tips for large-model downloads, GPU/CPU flags, and how the LocalGemma backend auto-pulls Ollama models when absent.

## Notes

- Keep the existing AST backend path intact as a non-LLM option for docstring generation.
- Preserve the llama-index-shaped response metadata (`raw.usage`) to minimize downstream churn while migrating callers to the new interface.
