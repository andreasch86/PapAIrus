import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import requests
from llama_index.core import Document, StorageContext, VectorStoreIndex, get_response_synthesizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import ConfigDict

from papairus.exceptions import EmbeddingServiceError, MissingEmbeddingModelError
from papairus.log import logger


def _extract_doc_text(doc: Document) -> str:
    """Return the document text without raising attribute errors."""

    # Primary path: the `text` attribute is standard on llama-index Documents.
    text_value = getattr(doc, "text", None)
    if text_value:
        return text_value

    # Fallback: if a helper exists, call it defensively.
    get_text = getattr(doc, "get_text", None)
    if callable(get_text):
        try:
            return get_text() or ""
        except Exception:
            return ""

    return ""


_DEFAULT_EMBED_BATCH_SIZE = 32
_MAX_EMBED_CHARS = 1024


def _raise_embedding_model_error(exc: Exception, embed_model) -> None:
    """Translate Ollama embedding errors into a clear ClickException."""

    if isinstance(exc, (MissingEmbeddingModelError, EmbeddingServiceError)):
        raise exc

    model_name = getattr(embed_model, "model_name", None) or getattr(
        embed_model, "model", "nomic-embed-text"
    )
    base_url = getattr(embed_model, "base_url", None)
    message = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    message = message or str(getattr(getattr(exc, "response", None), "text", ""))
    normalized_model_name = _normalize_model_name(model_name) or model_name

    if (status == 404 or "not found" in message) and "model" in message:
        details = (
            f"Ollama embedding model '{model_name}' is not available. "
            f"Run `ollama pull {model_name}`"
        )
        if base_url:
            details += f" against base URL {base_url}"
        details += " before retrying chat-with-repo."
        raise MissingEmbeddingModelError(details) from exc

    connection_error_markers = (
        "connection refused",
        "failed to establish a new connection",
        "connection reset",
        "eof",
        "timeout",
    )
    if status in {500, 502, 503} or any(marker in message for marker in connection_error_markers):
        target = base_url or "the configured Ollama host"
        raise EmbeddingServiceError(
            "Embedding request to {target} failed for model '{model}'. "
            "Ensure the Ollama daemon is reachable and the embedding model is healthy.".format(
                target=target, model=normalized_model_name
            )
        ) from exc

    return None


def _normalize_model_name(name: str | None) -> str | None:
    """Return the base model name without Ollama tag suffixes."""

    if not name:
        return None

    # Ollama tags come back as "model:tag"; the embed_model uses the base name.
    return name.split(":", 1)[0]


def _list_ollama_models(base_url: str | None) -> list[str] | None:
    """Return available Ollama model names or None if the endpoint is unreachable."""

    if not base_url:
        return None

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except Exception:
        return None

    payload = response.json()
    models = payload.get("models", []) if isinstance(payload, dict) else []

    normalized = {
        _normalize_model_name(entry.get("name")) for entry in models if isinstance(entry, dict)
    }
    return sorted(model for model in normalized if model)


def _validate_embedding_vector(embedding, base_url: str | None, model_name: str | None):
    """Ensure embeddings are non-empty numeric vectors."""

    if not isinstance(embedding, (list, tuple)):
        raise EmbeddingServiceError(
            "Embedding request to {url} failed for model '{model}'. "
            "Ensure the Ollama daemon is reachable and the embedding model is healthy.".format(
                url=base_url or "the configured Ollama host", model=model_name
            )
        )

    if len(embedding) == 0:
        raise EmbeddingServiceError(
            "Embedding request to {url} failed for model '{model}'. "
            "Received an empty embedding payload from the service.".format(
                url=base_url or "the configured Ollama host", model=model_name
            )
        )

    cleaned: list[float] = []
    for value in embedding:
        if not isinstance(value, (int, float)):
            raise EmbeddingServiceError(
                "Embedding request to {url} failed for model '{model}'. "
                "Embedding vector contained non-numeric values.".format(
                    url=base_url or "the configured Ollama host", model=model_name
                )
            )
        cleaned.append(float(value))

    return cleaned


def _build_repo_system_prompt(meta_data: list[dict]) -> str:
    """Construct a concise system prompt capturing repository context."""

    normalized_meta: list[dict] = []
    for entry in meta_data:
        if isinstance(entry, dict):
            normalized_meta.append(entry)
        else:
            normalized_meta.append({"source": str(entry), "type": "UnknownType"})

    total_items = len(normalized_meta)
    types = {str(entry.get("type", "UnknownType")) for entry in normalized_meta}
    sample_lines: list[str] = []

    for entry in normalized_meta[:5]:
        name = (
            entry.get("name")
            or entry.get("source")
            or entry.get("path")
            or entry.get("type")
            or "item"
        )
        entry_type = entry.get("type") or "UnknownType"
        location = entry.get("source") or entry.get("path")
        descriptor = f"- {name} ({entry_type})"
        if location:
            descriptor += f" from {location}"
        sample_lines.append(descriptor)

    header = (
        "You are an AI assistant answering questions about this repository. Use only the "
        "retrieved context chunks from the vector store when forming answers. If the "
        "context is insufficient, say you do not know rather than guessing."
    )

    summary = (
        f"Indexed repository summary: {total_items} items. "
        f"Item types include: {', '.join(sorted(types)) if types else 'unknown'}."
    )

    examples = "\n".join(sample_lines)
    if examples:
        examples = "Examples of indexed items:\n" + examples

    return "\n\n".join(part for part in [header, summary, examples] if part)


def _apply_system_prompt(llm, prompt: str) -> None:
    """Attach a system prompt to the LLM if the interface supports it."""

    if hasattr(llm, "update_system_prompt"):
        try:
            llm.update_system_prompt(prompt)
            return
        except Exception:
            pass

    if hasattr(llm, "system_prompt"):
        try:
            setattr(llm, "system_prompt", prompt)
            return
        except Exception:
            pass

    if hasattr(llm, "__dict__"):
        llm.__dict__["system_prompt"] = prompt


class _ChunkingEmbeddingWrapper(BaseEmbedding):
    """A light wrapper that chunks embedding batches to reduce request load."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, embed_model, max_batch_size: int = _DEFAULT_EMBED_BATCH_SIZE):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")

        object.__setattr__(self, "_wrapped_embed_model", embed_model)
        object.__setattr__(self, "_max_batch_size", max_batch_size)
        object.__setattr__(self, "_last_embedding_dim", None)

        model_name = getattr(embed_model, "model_name", None) or getattr(
            embed_model, "model", "unknown"
        )
        base_url = getattr(embed_model, "base_url", None)
        object.__setattr__(self, "base_url", base_url)

        try:
            super().__init__(model_name=model_name, embed_batch_size=max_batch_size)
        except TypeError:
            # Stubbed BaseEmbedding in tests may not accept keyword arguments.
            super().__init__()

        # Re-assign after BaseEmbedding init to avoid any BaseModel attribute handling
        # from stripping the internal reference.
        object.__setattr__(self, "_wrapped_embed_model", embed_model)
        object.__setattr__(self, "_max_batch_size", max_batch_size)

        # Mirror commonly inspected attributes used for logging and error handling.
        for attr in ("model_name", "model", "base_url"):
            if hasattr(embed_model, attr):
                object.__setattr__(self, attr, getattr(embed_model, attr))

    def _get_text_embedding(self, text: str):
        wrapped = object.__getattribute__(self, "_wrapped_embed_model")

        try:
            if hasattr(wrapped, "get_text_embedding"):
                embedding = wrapped.get_text_embedding(text)
                cleaned = _validate_embedding_vector(
                    embedding,
                    getattr(self, "base_url", None),
                    getattr(self, "model_name", None),
                )
                object.__setattr__(self, "_last_embedding_dim", len(cleaned))
                return cleaned
        except Exception:
            # Fall back to the HTTP embeddings endpoint when the wrapped model fails
            # (e.g., incompatible client or API change).
            return self._embed_via_http(text)

        raise AttributeError("Underlying embed model does not support embedding APIs")

    def _get_query_embedding(self, query: str):
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str):
        import asyncio

        return await asyncio.to_thread(self._get_query_embedding, query)

    async def aget_query_embedding(self, query: str):
        return await self._aget_query_embedding(query)

    def get_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    def get_text_embedding_batch(self, texts, show_progress: bool | None = None, **kwargs):
        # LlamaIndex passes a `show_progress` flag; ignore it for compatibility with
        # embedding clients that do not accept the parameter.
        _ = show_progress, kwargs
        return self._get_text_embeddings(texts)

    def get_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts):
        wrapped = object.__getattribute__(self, "_wrapped_embed_model")

        if hasattr(wrapped, "get_text_embedding_batch"):
            embeddings = []
            text_list = list(texts)
            for start in range(0, len(text_list), self._max_batch_size):
                batch = text_list[start : start + self._max_batch_size]
                try:
                    batch_embeddings = wrapped.get_text_embedding_batch(batch)
                    source = "wrapped"
                except Exception:
                    try:
                        batch_embeddings = self._embed_batch_via_http(batch)
                        source = "http"
                    except Exception as http_exc:
                        raise EmbeddingServiceError(str(http_exc)) from http_exc

                if not batch_embeddings or len(batch_embeddings) != len(batch):
                    if source != "http":
                        batch_embeddings = self._embed_batch_via_http(batch)
                    if not batch_embeddings or len(batch_embeddings) != len(batch):
                        raise EmbeddingServiceError(
                            "Embedding request to {url} failed for model '{model}'. "
                            "Received an empty embedding payload from the service.".format(
                                url=getattr(self, "base_url", None) or "the configured Ollama host",
                                model=getattr(self, "model_name", None),
                            )
                        )

                for emb in batch_embeddings:
                    cleaned = _validate_embedding_vector(
                        emb, getattr(self, "base_url", None), getattr(self, "model_name", None)
                    )
                    object.__setattr__(self, "_last_embedding_dim", len(cleaned))
                    embeddings.append(cleaned)

            return embeddings

        if hasattr(wrapped, "get_text_embedding"):
            return [
                _validate_embedding_vector(
                    self._get_text_embedding(text),
                    getattr(self, "base_url", None),
                    getattr(self, "model_name", None),
                )
                for text in texts
            ]

        raise AttributeError("Underlying embed model does not support embedding APIs")

    def __getattr__(self, name):
        # Delegate any other attributes to the wrapped embed model.
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        wrapped = object.__getattribute__(self, "_wrapped_embed_model")
        return getattr(wrapped, name)

    def __setattr__(self, name, value):
        # Allow tests to monkeypatch private attributes (e.g., HTTP fallback) while
        # keeping pydantic's BaseModel attribute validation for public fields.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        super().__setattr__(name, value)

    # HTTP fallback helpers -------------------------------------------------

    def _embed_via_http(self, text: str):
        base_url = getattr(self, "base_url", None)
        model_name = getattr(self, "model_name", None) or getattr(self, "model", None)

        # Trim extremely long payloads defensively to reduce the odds of empty or
        # failed embeddings from the server while preserving semantics.
        text = text[:_MAX_EMBED_CHARS]

        if not base_url or not model_name:
            raise EmbeddingServiceError(
                "Embedding request failed because base URL or model name was not configured."
            )

        payloads = [
            {"model": model_name, "prompt": text},
            {"model": model_name, "input": text},
            {"model": model_name, "input": [text]},
        ]
        last_exc: Exception | None = None

        try:
            import ollama

            client = ollama.Client(host=base_url)
            client_payload = {"model": model_name, "prompt": text}
            client_response = client.embeddings(**client_payload)
            if isinstance(client_response, dict) and "embedding" in client_response:
                cleaned = _validate_embedding_vector(
                    client_response["embedding"], base_url, model_name
                )
                object.__setattr__(self, "_last_embedding_dim", len(cleaned))
                return cleaned
        except Exception as exc:  # noqa: BLE001 - fall back to raw HTTP below
            last_exc = exc

        def _extract_embedding(payload: dict):
            if not isinstance(payload, dict):
                raise EmbeddingServiceError(
                    "Embedding request to {url} failed for model '{model}'. "
                    "Ensure the Ollama daemon is reachable and the embedding model is healthy.".format(
                        url=base_url, model=model_name
                    )
                )

            if "embedding" in payload:
                return _validate_embedding_vector(payload["embedding"], base_url, model_name)

            if payload.get("data"):
                first = payload["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return _validate_embedding_vector(first["embedding"], base_url, model_name)

            if payload.get("embeddings"):
                first = payload["embeddings"][0]
                if isinstance(first, (list, tuple)):
                    return _validate_embedding_vector(first, base_url, model_name)

            raise EmbeddingServiceError(
                "Embedding request to {url} failed for model '{model}'. "
                "Ensure the Ollama daemon is reachable and the embedding model is healthy.".format(
                    url=base_url, model=model_name
                )
            )

        for payload in payloads:
            try:
                response = requests.post(f"{base_url}/api/embeddings", json=payload, timeout=60)
                response.raise_for_status()

                try:
                    cleaned = _extract_embedding(response.json())
                    object.__setattr__(self, "_last_embedding_dim", len(cleaned))
                    return cleaned
                except EmbeddingServiceError as embed_exc:
                    last_exc = embed_exc
                    # Try the alternate payload when the payload shape was accepted
                    # but produced an invalid embedding (e.g., empty payload).
                    continue
            except Exception as exc:  # noqa: BLE001 - inspected below
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                # Try the alternate payload only for server-side failures.
                if status and status >= 500:
                    continue
                _raise_embedding_model_error(exc, self)

        if (
            isinstance(last_exc, EmbeddingServiceError)
            and "empty embedding payload" in str(last_exc).lower()
            and getattr(self, "_last_embedding_dim", None)
        ):
            dim = int(self._last_embedding_dim)  # type: ignore[arg-type]
            logger.warning(
                "Embedding service returned an empty vector; substituting zeros of length %s.",
                dim,
            )
            return [0.0 for _ in range(dim)]

        _raise_embedding_model_error(
            last_exc
            or EmbeddingServiceError(
                "Embedding request to {url} failed for model '{model}'. "
                "Ensure the Ollama daemon is reachable and the embedding model is healthy.".format(
                    url=base_url, model=model_name
                )
            ),
            self,
        )

    def _embed_batch_via_http(self, texts):
        return [self._embed_via_http(text) for text in texts]


def _wrap_chunking_embed_model(embed_model, max_batch_size: int | None = None):
    """Wrap embed_model with chunked batching unless already wrapped."""

    if isinstance(embed_model, _ChunkingEmbeddingWrapper):
        return embed_model

    if max_batch_size is not None and max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1")

    batch_size = max_batch_size or _DEFAULT_EMBED_BATCH_SIZE
    return _ChunkingEmbeddingWrapper(embed_model, batch_size)


def _ensure_embedding_model_available(embed_model) -> None:
    """Validate the configured embedding model is pullable before indexing."""

    try:
        model_name = getattr(embed_model, "model_name", None) or getattr(
            embed_model, "model", "nomic-embed-text"
        )
        normalized_model_name = _normalize_model_name(model_name) or model_name
        base_url = getattr(embed_model, "base_url", None)

        available_models = _list_ollama_models(base_url)
        if base_url and available_models is None:
            raise EmbeddingServiceError(
                "Could not reach Ollama at {url}. Ensure the daemon is running,"
                " accessible, and that the host/port matches the chat-with-repo"
                " configuration.".format(url=base_url)
            )

        if available_models is not None and normalized_model_name not in available_models:
            raise MissingEmbeddingModelError(
                "Ollama embedding model '{model}' is not available at {url}. "
                "Run `ollama pull {model}` and retry chat-with-repo.".format(
                    model=normalized_model_name, url=base_url
                )
            )

        if hasattr(embed_model, "get_text_embedding"):
            embed_model.get_text_embedding("__healthcheck__")
        elif hasattr(embed_model, "get_text_embedding_batch"):
            embed_model.get_text_embedding_batch(["__healthcheck__"])
    except Exception as exc:  # noqa: BLE001 - propagate meaningful guidance
        _raise_embedding_model_error(exc, embed_model)
        raise


def _get_node_content(node) -> str:
    """Best-effort extraction of node text content for sizing decisions."""

    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        try:
            return get_content(metadata_mode="none") or ""
        except Exception:
            return ""

    text_value = getattr(node, "text", None)
    if text_value:
        return text_value

    if isinstance(node, str):
        return node

    return ""


def _rechunk_oversized_nodes(nodes, max_chars: int = _MAX_EMBED_CHARS):
    """Ensure nodes stay under embedding-safe sizes by re-splitting long chunks."""

    if not nodes:
        return []

    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    balanced_nodes = []
    queue = list(nodes)

    while queue:
        node = queue.pop(0)
        content = _get_node_content(node)
        if len(content) <= max_chars:
            balanced_nodes.append(node)
            continue

        meta = getattr(node, "metadata", None) or getattr(node, "extra_info", None)
        safe_chunk_size = max(max_chars, len(str(meta)) + 1 if meta is not None else max_chars)
        doc = Document(text=content, extra_info=meta if isinstance(meta, dict) else None)
        splitter = SentenceSplitter(chunk_size=safe_chunk_size, chunk_overlap=safe_chunk_size // 10)
        new_nodes = splitter.get_nodes_from_documents([doc])

        if not new_nodes or all(
            len(_get_node_content(new_node)) >= len(content) for new_node in new_nodes
        ):
            # The splitter returned nothing or failed to reduce the size; fall back to a
            # deterministic slicing approach to avoid re-queuing the same oversized node
            # repeatedly.
            new_nodes = [
                Document(
                    text=content[i : i + max_chars],
                    extra_info=meta if isinstance(meta, dict) else None,
                )
                for i in range(0, len(content), max_chars)
            ]
            logger.debug(
                f"Fallback manual rechunking split oversized node of length {len(content)} into {len(new_nodes)} fixed chunks."
            )
        else:
            logger.debug(
                f"Re-splitting oversized node of length {len(content)} into {len(new_nodes)} chunks for embedding safety."
            )
        queue = list(new_nodes) + queue

    return balanced_nodes


class VectorStoreManager:
    def __init__(
        self,
        top_k,
        llm,
        embed_model,
        embed_batch_size: int | None = None,
        max_workers: int | None = None,
    ):
        """
        Initialize the VectorStoreManager.
        """
        self.query_engine = None  # Initialize as None
        self.chroma_db_path = "./chroma_db"  # Path to Chroma database
        self.collection_name = "test"  # Default collection name
        self.similarity_top_k = top_k
        self.llm = llm
        self._base_embed_model = embed_model
        self.embed_model = _wrap_chunking_embed_model(embed_model, embed_batch_size)
        self.max_workers = max_workers

    def _current_batch_size(self) -> int:
        return int(
            getattr(self.embed_model, "_max_batch_size", None)
            or getattr(self.embed_model, "embed_batch_size", 1)
        )

    def _reset_embed_model_batch(self, batch_size: int) -> None:
        base_model = getattr(self.embed_model, "_wrapped_embed_model", self._base_embed_model)
        self.embed_model = _wrap_chunking_embed_model(base_model, batch_size)
        _ensure_embedding_model_available(self.embed_model)

    def _process_document(self, index: int, doc: Document, splitter_factory):
        text_content = _extract_doc_text(doc)
        logger.debug(f"Processing document {index + 1}: Content length={len(text_content)}")

        splitter = splitter_factory()
        try:
            nodes = splitter.get_nodes_from_documents([doc])
            logger.debug(f"Document {index + 1} split into {len(nodes)} semantic chunks.")
        except Exception as exc:  # noqa: BLE001 - fall back for splitter failures
            logger.warning(
                "Semantic splitting failed for document %s, falling back to SentenceSplitter. Error: %s",
                index + 1,
                exc,
            )

            safe_chunk_size = max(1024, len(str(getattr(doc, "extra_info", ""))) + 1)
            logger.debug(
                "Using SentenceSplitter chunk_size=%s for document %s to accommodate metadata of length %s.",
                safe_chunk_size,
                index + 1,
                len(str(getattr(doc, "extra_info", ""))),
            )
            base_splitter = SentenceSplitter(chunk_size=safe_chunk_size)
            nodes = base_splitter.get_nodes_from_documents([doc])
            logger.debug(f"Document {index + 1} split into {len(nodes)} sentence chunks.")

        filtered = [node for node in nodes if _get_node_content(node).strip()]
        dropped = len(nodes) - len(filtered)
        if dropped:
            logger.debug(
                "Dropped %s empty chunks for document %s before indexing.", dropped, index + 1
            )

        return filtered

    def create_vector_store(self, md_contents, meta_data):
        """
        Add markdown content and metadata to the index.
        """
        if not md_contents or not meta_data:
            logger.warning("No content or metadata provided. Skipping.")
            return

        # Ensure lengths match
        min_length = min(len(md_contents), len(meta_data))
        md_contents = md_contents[:min_length]
        meta_data = meta_data[:min_length]

        logger.debug(f"Number of markdown contents: {len(md_contents)}")
        logger.debug(f"Number of metadata entries: {len(meta_data)}")

        repo_system_prompt = _build_repo_system_prompt(meta_data)
        _apply_system_prompt(self.llm, repo_system_prompt)

        # Fail fast if the Ollama embedding model is unavailable, instead of
        # streaming a series of semantic-splitter fallbacks before the eventual
        # embed call fails during index construction.
        _ensure_embedding_model_available(self.embed_model)

        # Initialize Chroma client and collection
        db = chromadb.PersistentClient(path=self.chroma_db_path)
        chroma_collection = db.get_or_create_collection(self.collection_name)

        existing_ids = set(chroma_collection.get()["ids"])

        documents_to_process = []

        for content, meta in zip(md_contents, meta_data):
            doc_hash = hashlib.sha256((content + str(meta)).encode("utf-8")).hexdigest()
            # We use doc_hash as the document ID
            if doc_hash not in existing_ids:
                 doc = Document(text=content, extra_info=meta)
                 doc.id_ = doc_hash
                 documents_to_process.append(doc)

        logger.info(f"Found {len(documents_to_process)} new documents to index out of {len(md_contents)}.")

        # Initialize semantic chunker (SimpleNodeParser)
        logger.debug("Initializing semantic chunker (SimpleNodeParser).")
        splitter_factory = lambda: SemanticSplitterNodeParser(  # noqa: E731
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model,
        )

        all_nodes = []
        if documents_to_process:
            worker_count = self.max_workers or os.cpu_count() or 4
            worker_count = min(max(1, worker_count), len(documents_to_process))
            logger.debug("Processing documents with ThreadPoolExecutor(max_workers=%s).", worker_count)

            results: dict[int, list] = {}
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(self._process_document, idx, doc, splitter_factory): idx
                    for idx, doc in enumerate(documents_to_process)
                }

                for future in as_completed(futures):
                    doc_index = futures[future]
                    try:
                        results[doc_index] = future.result()
                    except Exception as exc:  # noqa: BLE001 - propagate for visibility
                        logger.exception(
                            "Document %s failed during splitting; aborting vector store creation.",
                            doc_index + 1,
                        )
                        raise exc

            all_nodes = [node for idx in sorted(results) for node in results[idx]]

            all_nodes = _rechunk_oversized_nodes(all_nodes)

            all_nodes = [node for node in all_nodes if _get_node_content(node).strip()]

        if documents_to_process and not all_nodes:
            logger.warning("No valid nodes to add to the index after chunking.")
            if not existing_ids:
                return
            # Continue to load existing index
            documents_to_process = []

        logger.debug(f"Number of valid chunks: {len(all_nodes)}")

        # Set up ChromaVectorStore and load data
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if documents_to_process:
            while True:
                try:
                    index = VectorStoreIndex(
                        all_nodes, storage_context=storage_context, embed_model=self.embed_model
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - surface friendly guidance
                    try:
                        _raise_embedding_model_error(exc, self.embed_model)
                    except MissingEmbeddingModelError:
                        raise
                    except EmbeddingServiceError as embed_exc:
                        if self._current_batch_size() > 1:
                            logger.warning(
                                "Embedding failed; retrying with embed batch size=1 to reduce payload."
                            )
                            self._reset_embed_model_batch(batch_size=1)
                            continue
                        raise embed_exc
                    raise
        else:
             index = VectorStoreIndex.from_vector_store(
                 vector_store=vector_store, embed_model=self.embed_model
             )

        retriever = VectorIndexRetriever(
            index=index, similarity_top_k=self.similarity_top_k, embed_model=self.embed_model
        )

        response_synthesizer = get_response_synthesizer(llm=self.llm)

        # Set the query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        logger.info(f"Vector store created and loaded with {len(documents_to_process)} new documents.")

    def query_store(self, query):
        """
        Query the vector store for relevant documents.
        """
        if not self.query_engine:
            logger.error("Query engine is not initialized. Please create a vector store first.")
            return []

        # Query the vector store
        logger.debug(f"Querying vector store with: {query}")
        results = self.query_engine.query(query)

        # Extract relevant information from results
        return [{"text": results.response, "metadata": results.metadata}]
