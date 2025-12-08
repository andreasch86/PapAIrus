import re

from papairus.chat_with_repo.json_handler import JsonFileProcessor
from papairus.chat_with_repo.prompt import query_generation_template, rag_ar_template, rag_template
from papairus.chat_with_repo.text_analysis_tool import TextAnalysisTool
from papairus.chat_with_repo.vector_store_manager import VectorStoreManager
from papairus.llm_provider import build_embedding_model, build_llm
from papairus.log import logger


class RepoAssistant:
    def __init__(self, chat_settings, db_path):
        self.db_path = db_path
        self.md_contents = []

        self.weak_model = build_llm(chat_settings)
        self.strong_model = build_llm(chat_settings)
        self.textanslys = TextAnalysisTool(self.weak_model, db_path)
        self.json_data = JsonFileProcessor(db_path)
        embed_model = build_embedding_model(chat_settings)
        self.vector_store_manager = VectorStoreManager(
            top_k=5, llm=self.weak_model, embed_model=embed_model
        )

    def generate_queries(self, query_str: str, num_queries: int = 4):
        fmt_prompt = query_generation_template.format(num_queries=num_queries - 1, query=query_str)
        response = self.weak_model.complete(fmt_prompt)
        raw_queries = response.text.split("\n") if hasattr(response, "text") else []
        return self._sanitize_generated_queries(raw_queries)

    def _sanitize_generated_queries(self, queries: list[str]):
        cleaned_queries: list[str] = []
        skip_prefixes = ("**query", "query", "sure,", "here are", "**", "-")

        for query in queries:
            if not isinstance(query, str):
                continue

            normalized = query.strip().strip("`").strip()
            if not normalized:
                continue

            lowered = normalized.lower()
            if lowered.startswith(skip_prefixes):
                continue

            cleaned_queries.append(normalized)

        if not cleaned_queries:
            logger.debug("No usable queries were generated from the model response.")

        return cleaned_queries

    def rerank(self, query, docs):
        if not docs:
            return []

        query_terms = [term for term in re.split(r"\W+", str(query).lower()) if term]
        if not query_terms:
            return list(docs)[:5]

        scored: list[tuple[int, int, str]] = []
        for idx, doc in enumerate(docs):
            text = str(doc)
            lowered = text.lower()
            score = sum(lowered.count(term) for term in query_terms)
            scored.append((score, idx, text))

        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [doc for score, _idx, doc in scored if doc][:5]
        return ranked or list(docs)[:5]

    def rag(self, query, retrieved_documents):
        rag_prompt = rag_template.format(query=query, information="\n\n".join(retrieved_documents))
        response = self.weak_model.complete(rag_prompt)
        return response.text

    def list_to_markdown(self, list_items):
        markdown_content = ""

        for index, item in enumerate(list_items, start=1):
            markdown_content += f"{index}. {item}\n"

        return markdown_content

    def rag_ar(self, query, related_code, embedding_recall, project_name):
        rag_ar_prompt = rag_ar_template.format_messages(
            query=query,
            related_code=related_code,
            embedding_recall=embedding_recall,
            project_name=project_name,
        )
        response = self.strong_model.chat(rag_ar_prompt)
        return response.message.content

    def respond(self, message, instruction):
        """
        Respond to a user query by processing input, querying the vector store,
        reranking results, and generating a final response.
        """
        logger.debug("Starting response generation.")

        # Step 1: Format the chat prompt
        prompt = self.textanslys.format_chat_prompt(message, instruction)
        logger.debug(f"Formatted prompt: {prompt}")

        keyword_response = self.textanslys.keyword(prompt)
        questions = getattr(keyword_response, "text", str(keyword_response))
        logger.debug(f"Generated keywords from prompt: {questions}")

        # Step 2: Generate additional queries
        generated_queries = [
            query.strip() for query in self.generate_queries(prompt, 3) if query.strip()
        ]
        logger.debug(f"Generated queries: {generated_queries}")

        # Always include the user's original message as the first query to anchor
        # retrieval on the actual ask. Deduplicate while preserving order.
        prompt_queries: list[str] = []
        seen: set[str] = set()
        for candidate in [str(message)] + generated_queries:
            cleaned = candidate.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            prompt_queries.append(cleaned)

        all_results = []

        # Step 3: Query the VectorStoreManager for each query
        for query in prompt_queries:
            cleaned_query = query.strip()
            logger.debug(f"Querying vector store with: {cleaned_query}")
            query_results = self.vector_store_manager.query_store(cleaned_query)
            logger.debug(f"Results for query '{query}': {query_results}")
            all_results.extend(query_results)

        if not all_results:
            fallback = (
                "I could not find any relevant information in the repository for this question. "
                "Please try rephrasing or ask about a specific file or function."
            )
            logger.debug("No vector search results found; returning fallback response.")
            return message, fallback, "", str(questions), "", ""

        # Step 4: Deduplicate results by content
        unique_results = {result["text"]: result for result in all_results}.values()
        unique_documents = [result["text"] for result in unique_results]
        logger.debug(f"Unique documents: {unique_documents}")

        unique_code = [
            result.get("metadata", {}).get("code_content")
            for result in unique_results
            if result.get("metadata", {}).get("code_content")
        ]
        logger.debug(f"Unique code content: {unique_code}")

        # Step 5: Rerank documents based on relevance
        retrieved_documents = self.rerank(message, unique_documents)
        logger.debug(f"Reranked documents: {retrieved_documents}")

        # Step 6: Generate a response using RAG (Retrieve and Generate)
        response = self.rag(prompt, retrieved_documents)
        chunkrecall = self.list_to_markdown(retrieved_documents)
        logger.debug(f"RAG-generated response: {response}")
        logger.debug(f"Markdown chunk recall: {chunkrecall}")

        bot_message = str(response)
        logger.debug(f"Initial bot_message: {bot_message}")

        # Step 7: Perform NER and queryblock processing
        keyword = str(self.textanslys.nerquery(bot_message))
        keywords = str(self.textanslys.nerquery(str(prompt) + str(questions)))
        logger.debug(f"Extracted keywords: {keyword}, {keywords}")

        codez, mdz = self.textanslys.queryblock(keyword)
        codey, mdy = self.textanslys.queryblock(keywords)

        # Ensure all returned items are lists
        codez = codez if isinstance(codez, list) else [codez]
        mdz = mdz if isinstance(mdz, list) else [mdz]
        codey = codey if isinstance(codey, list) else [codey]
        mdy = mdy if isinstance(mdy, list) else [mdy]

        # Step 8: Merge and deduplicate results
        codex = list(dict.fromkeys(codez + codey))
        md = list(
            dict.fromkeys(tuple(item) if isinstance(item, list) else item for item in mdz + mdy)
        )

        flattened_md: list[str] = []
        for item in md:
            if isinstance(item, tuple):
                flattened_md.extend([str(val) for val in item if val])
            elif isinstance(item, str):
                if item:
                    flattened_md.append(item)

        uni_codex = list(dict.fromkeys(codex))
        uni_md = list(dict.fromkeys(flattened_md))

        # Convert to Markdown format
        codex_md = self.textanslys.list_to_markdown(uni_codex)
        retrieved_documents = list(dict.fromkeys(retrieved_documents + uni_md))

        # Final response generation using top-ranked documents and code
        retrieved_documents = retrieved_documents[:6]
        uni_code = list(dict.fromkeys(uni_codex + unique_code))[:6]

        unique_code_md = self.textanslys.list_to_markdown(unique_code)
        logger.debug(f"Unique code in Markdown: {unique_code_md}")

        # Generate final response using RAG_AR
        bot_message = self.rag_ar(prompt, uni_code, retrieved_documents, "test")
        logger.debug(f"Final bot_message after RAG_AR: {bot_message}")

        return message, bot_message, chunkrecall, questions, unique_code_md, codex_md
