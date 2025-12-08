import os
import sys
import types
from pathlib import Path

import pytest

try:
    import git
except ImportError:  # pragma: no cover - optional dependency for a subset of tests
    git = None


def _install_stub_dependencies() -> None:
    """Provide light stub modules when optional heavy deps are unavailable."""

    if "chromadb" not in sys.modules:
        chromadb = types.ModuleType("chromadb")

        class _PersistentClient:  # pragma: no cover - stubbed test helper
            def __init__(self, *_, **__):
                pass

        chromadb.PersistentClient = _PersistentClient
        sys.modules["chromadb"] = chromadb

    if "llama_index" not in sys.modules:
        llama_index = types.ModuleType("llama_index")
        sys.modules["llama_index"] = llama_index
    else:
        llama_index = sys.modules["llama_index"]

    if "llama_index.core" not in sys.modules:
        core = types.ModuleType("llama_index.core")
        sys.modules["llama_index.core"] = core
        llama_index.core = core
    else:
        core = sys.modules["llama_index.core"]

    class _Document:
        def __init__(self, text: str = "", metadata=None, extra_info=None, doc_id=None):
            self.text = text
            self.metadata = metadata or {}
            self.extra_info = extra_info or {}
            self.doc_id = doc_id

    class _PromptTemplate:
        def __init__(self, template: str):
            self.template = template

        @classmethod
        def from_template(cls, template):
            return cls(template)

        def format(self, **kwargs):
            return self.template.format(**kwargs)

    class _ChatPromptTemplate:
        def __init__(self, message_templates=None, **_kwargs):
            self.message_templates = message_templates or []

        @classmethod
        def from_messages(cls, messages):
            return cls(messages)

        def format_messages(self, **_kwargs):  # pragma: no cover - stub
            return self.message_templates

    class _StorageContext:
        @classmethod
        def from_defaults(cls, **_kwargs):
            return cls()

    class _VectorStoreIndex:
        @classmethod
        def from_documents(cls, *_, **__):
            return types.SimpleNamespace(as_retriever=lambda **_: None)

    def _response_synthesizer(llm=None):  # pragma: no cover - stubbed
        return None

    core.Document = _Document
    core.StorageContext = _StorageContext
    core.VectorStoreIndex = _VectorStoreIndex
    core.get_response_synthesizer = _response_synthesizer
    core.PromptTemplate = _PromptTemplate
    core.ChatPromptTemplate = _ChatPromptTemplate

    node_parser = types.ModuleType("llama_index.core.node_parser")
    sys.modules["llama_index.core.node_parser"] = node_parser

    class _SemanticSplitterNodeParser:
        def __init__(self, *_, **__):
            pass

        def get_nodes_from_documents(self, docs):
            return docs

    class _SentenceSplitter:
        def __init__(self, *_, **__):
            pass

        def get_nodes_from_documents(self, docs):
            return docs

    node_parser.SemanticSplitterNodeParser = _SemanticSplitterNodeParser
    node_parser.SentenceSplitter = _SentenceSplitter
    core.node_parser = node_parser

    query_engine = types.ModuleType("llama_index.core.query_engine")
    sys.modules["llama_index.core.query_engine"] = query_engine

    class _RetrieverQueryEngine:
        def __init__(self, retriever=None, response_synthesizer=None):
            self.retriever = retriever
            self.response_synthesizer = response_synthesizer

        def query(self, query):  # pragma: no cover - stubbed
            return types.SimpleNamespace(response="", metadata={})

    query_engine.RetrieverQueryEngine = _RetrieverQueryEngine
    core.query_engine = query_engine

    retrievers = types.ModuleType("llama_index.core.retrievers")
    sys.modules["llama_index.core.retrievers"] = retrievers

    class _VectorIndexRetriever:
        def __init__(self, *_, **__):
            pass

    retrievers.VectorIndexRetriever = _VectorIndexRetriever
    core.retrievers = retrievers

    embeddings_base = types.ModuleType("llama_index.core.base.embeddings.base")
    sys.modules["llama_index.core.base.embeddings.base"] = embeddings_base

    class _BaseEmbedding:
        def __init__(self, *_, model_name=None, embed_batch_size=None, **__):
            self.model_name = model_name
            self.embed_batch_size = embed_batch_size

        def class_name(self):
            return "BaseEmbedding"

        def get_query_embedding(self, *_):
            return []

    embeddings_base.BaseEmbedding = _BaseEmbedding

    vector_stores = types.ModuleType("llama_index.vector_stores")
    sys.modules["llama_index.vector_stores"] = vector_stores

    chroma_module = types.ModuleType("llama_index.vector_stores.chroma")
    sys.modules["llama_index.vector_stores.chroma"] = chroma_module

    class _ChromaVectorStore:
        def __init__(self, *_, **__):
            pass

    chroma_module.ChromaVectorStore = _ChromaVectorStore
    vector_stores.chroma = chroma_module

    llms = types.ModuleType("llama_index.core.llms")
    sys.modules["llama_index.core.llms"] = llms

    function_calling = types.ModuleType("llama_index.core.llms.function_calling")
    sys.modules["llama_index.core.llms.function_calling"] = function_calling

    class _FunctionCallingLLM:  # pragma: no cover - stub
        def complete(self, *_args, **_kwargs):
            return types.SimpleNamespace(text="")

        def chat(self, *_args, **_kwargs):
            return types.SimpleNamespace(message=types.SimpleNamespace(content=""))

    function_calling.FunctionCallingLLM = _FunctionCallingLLM
    function_calling.ChatMessage = types.SimpleNamespace

    class _MessageRole:
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"

    function_calling.MessageRole = _MessageRole
    llms.function_calling = function_calling
    llms.ChatMessage = types.SimpleNamespace
    llms.MessageRole = _MessageRole
    core.llms = llms

    if "gradio" not in sys.modules:
        gradio = types.ModuleType("gradio")
    else:
        gradio = sys.modules["gradio"]

    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, *_, **__):
            return False

        def queue(self):
            return self

        def launch(self, server_name: str = "127.0.0.1", server_port: int = 7860, **_kwargs):
            from http.server import BaseHTTPRequestHandler, HTTPServer
            import threading

            class _Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    """
                    Responds to HTTP GET requests with a 200 OK response.
                    
                    Args:
                        self: The HTTP server instance.
                    """
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def log_message(self, *_args, **_kwargs):  # pragma: no cover - quiet server
                    """
                    Logs an HTTP request message.
                    
                    Args:
                        self: The HTTP server instance.
                        *args: Positional arguments.
                        **kwargs: Keyword arguments.
                    
                    Returns:
                        None.
                    """
                    return None

            self._server = HTTPServer((server_name, server_port), _Handler)
            thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            thread.start()
            return types.SimpleNamespace(close=self.close)

        def close(self):
            """
            Shuts down the HTTP server.
            
            Args:
                self: The HTTP server instance.
            """
            server = getattr(self, "_server", None)
            if server:
                server.shutdown()

    class _Textbox:
        """
        A class representing a text input field.
        
        Attributes:
            value: The current value of the text input field.
        
        Methods:
            submit: Submits the text input field.
        """
        def __init__(self, *_, **__):
            self.value = ""

        def submit(self, *_, **__):
            """
            Submits a text input field.
            
            Args:
                self: The text input field instance.
            """
            return None

    class _Button:
        def __init__(self, *_, **__):
            pass

        def click(self, *_, **__):
            return None

    gradio.Blocks = lambda *a, **k: _Context()
    gradio.Markdown = lambda *a, **k: None
    gradio.Tab = lambda *a, **k: _Context()
    gradio.Row = lambda *a, **k: _Context()
    gradio.Column = lambda *a, **k: _Context()
    gradio.Textbox = _Textbox
    gradio.Button = _Button
    gradio.ClearButton = _Button
    gradio.HTML = lambda *a, **k: ""
    gradio.close_all = lambda: None
    gradio.IS_STUB = True

    sys.modules["gradio"] = gradio


_install_stub_dependencies()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from papairus.settings import (
    ChatCompletionSettings,
    LogLevel,
    ProjectSettings,
    Setting,
    SettingsManager,
)


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    SettingsManager._setting_instance = None
    yield
    SettingsManager._setting_instance = None


@pytest.fixture()
def temp_repo(tmp_path, monkeypatch):
    if git is None:
        pytest.skip("GitPython is required for repository-backed tests")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo = git.Repo.init(repo_path)
    (repo_path / "sample.py").write_text("print('hi')\n")
    repo.index.add(["sample.py"])
    repo.index.commit("init")

    project_settings = ProjectSettings(
        target_repo=repo_path,
        hierarchy_name=".project_doc_record",
        markdown_docs_name="docs",
        ignore_list=[],
        language="English (UK)",
        max_thread_count=1,
        log_level=LogLevel.INFO,
        telemetry_opt_in=False,
    )
    chat_settings = ChatCompletionSettings()
    SettingsManager._setting_instance = Setting(
        project=project_settings, chat_completion=chat_settings
    )
    return repo
