import hashlib
import time
from pathlib import Path

from papairus.chat_with_repo.gradio_interface import GradioInterface
from papairus.chat_with_repo.rag import RepoAssistant
from papairus.log import logger
from papairus.settings import ChatCompletionSettings, SettingsManager


def _select_repo_chat_settings(settings: ChatCompletionSettings) -> ChatCompletionSettings:
    """Pin chat-with-repo to the CodeGemma instruct model served by Ollama."""

    update = {"model": "codegemma", "ollama_model": "codegemma:7b-instruct-q4_K_M"}

    if settings.model != "codegemma" or settings.ollama_model != "codegemma:7b-instruct-q4_K_M":
        logger.info(
            "chat-with-repo supports only the Ollama CodeGemma instruct model; forcing configuration."
        )

    return settings.model_copy(update=update)


def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    logger.info("Initializing the PapAIrus chat with doc module.")

    # Load settings
    setting = SettingsManager.get_setting()
    chat_settings = _select_repo_chat_settings(setting.chat_completion)

    db_path = (
        setting.project.target_repo / setting.project.hierarchy_name / "project_hierarchy.json"
    )

    # Initialize RepoAssistant
    assistant = RepoAssistant(chat_settings, db_path)

    # Check for existing hash
    chroma_db_path = Path(assistant.vector_store_manager.chroma_db_path)
    hash_file = chroma_db_path / "vector_store_hash.txt"

    current_hash = get_file_hash(db_path) if db_path.exists() else ""

    rebuild = True
    if hash_file.exists() and current_hash:
        stored_hash = hash_file.read_text().strip()
        if stored_hash == current_hash:
            logger.info("Vector store is up to date. Skipping creation.")
            rebuild = False

    if rebuild:
        # Extract data
        md_contents, meta_data = assistant.json_data.extract_data()

        # Create vector store and measure runtime
        logger.info("Starting vector store creation...")
        start_time = time.time()
        assistant.vector_store_manager.create_vector_store(md_contents, meta_data)
        elapsed_time = time.time() - start_time
        logger.info(f"Vector store created successfully in {elapsed_time:.2f} seconds.")

        # Save hash
        if current_hash:
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            hash_file.write_text(current_hash)

    # Launch Gradio interface
    GradioInterface(assistant.respond)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
