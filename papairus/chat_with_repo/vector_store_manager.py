import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from papairus.exceptions import MissingEmbeddingModelError
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


def _raise_embedding_model_error(exc: Exception, embed_model) -> None:
    """Translate Ollama embedding errors into a clear ClickException."""

    model_name = getattr(embed_model, "model_name", None) or getattr(
        embed_model, "model", "nomic-embed-text"
    )
    base_url = getattr(embed_model, "base_url", None)
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)

    if (status == 404 or "not found" in message) and "model" in message:
        details = (
            f"Ollama embedding model '{model_name}' is not available. "
            f"Run `ollama pull {model_name}`"
        )
        if base_url:
            details += f" against base URL {base_url}"
        details += " before retrying chat-with-repo."
        raise MissingEmbeddingModelError(details) from exc


class VectorStoreManager:
    def __init__(self, top_k, llm, embed_model):
        """
        Initialize the VectorStoreManager.
        """
        self.query_engine = None  # Initialize as None
        self.chroma_db_path = "./chroma_db"  # Path to Chroma database
        self.collection_name = "test"  # Default collection name
        self.similarity_top_k = top_k
        self.llm = llm
        self.embed_model = embed_model

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

        # Initialize Chroma client and collection
        db = chromadb.PersistentClient(path=self.chroma_db_path)
        chroma_collection = db.get_or_create_collection(self.collection_name)

        # Initialize semantic chunker (SimpleNodeParser)
        logger.debug("Initializing semantic chunker (SimpleNodeParser).")
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model,
        )

        documents = [
            Document(text=content, extra_info=meta) for content, meta in zip(md_contents, meta_data)
        ]

        all_nodes = []
        for i, doc in enumerate(documents):
            text_content = _extract_doc_text(doc)
            logger.debug(
                f"Processing document {i+1}: Content length={len(text_content)}"
            )

            try:
                # Try semantic splitting first
                nodes = splitter.get_nodes_from_documents([doc])
                logger.debug(f"Document {i+1} split into {len(nodes)} semantic chunks.")

            except Exception as e:
                # Fallback to baseline sentence splitting
                logger.warning(
                    f"Semantic splitting failed for document {i+1}, falling back to SentenceSplitter. Error: {e}"
                )

                safe_chunk_size = max(1024, len(str(getattr(doc, "extra_info", ""))) + 1)
                logger.debug(
                    "Using SentenceSplitter chunk_size=%s for document %s to accommodate metadata of length %s.",
                    safe_chunk_size,
                    i + 1,
                    len(str(getattr(doc, "extra_info", ""))),
                )
                base_splitter = SentenceSplitter(chunk_size=safe_chunk_size)
                nodes = base_splitter.get_nodes_from_documents([doc])
                logger.debug(f"Document {i+1} split into {len(nodes)} sentence chunks.")

            all_nodes.extend(nodes)

        if not all_nodes:
            logger.warning("No valid nodes to add to the index after chunking.")
            return

        logger.debug(f"Number of valid chunks: {len(all_nodes)}")

        # Set up ChromaVectorStore and load data
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        try:
            index = VectorStoreIndex(
                all_nodes, storage_context=storage_context, embed_model=self.embed_model
            )
        except Exception as exc:  # noqa: BLE001 - surface friendly guidance
            _raise_embedding_model_error(exc, self.embed_model)
            raise
        retriever = VectorIndexRetriever(
            index=index, similarity_top_k=self.similarity_top_k, embed_model=self.embed_model
        )

        response_synthesizer = get_response_synthesizer(llm=self.llm)

        # Set the query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        logger.info(f"Vector store created and loaded with {len(documents)} documents.")

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
