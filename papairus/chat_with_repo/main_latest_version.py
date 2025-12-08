import time

from papairus.chat_with_repo.gradio_interface import GradioInterface
from papairus.chat_with_repo.rag import RepoAssistant
from papairus.log import logger
from papairus.settings import ChatCompletionSettings, SettingsManager


def _select_repo_chat_settings(settings: ChatCompletionSettings) -> ChatCompletionSettings:
    """
    Pin chat-with-repo to the CodeGemma instruct model served by Ollama.

    Args:
        settings: The current ChatCompletionSettings object.

    Returns:
        A new ChatCompletionSettings object with the model and ollama_model fields updated to the CodeGemma instruct model.
    """

    update = {"model": "codegemma", "ollama_model": "codegemma:7b-instruct-q4_K_M"}

    if settings.model != "codegemma" or settings.ollama_model != "codegemma:7b-instruct-q4_K_M":
        logger.info(
            "chat-with-repo supports only the Ollama CodeGemma instruct model; forcing configuration."
        )

    return settings.model_copy(update=update)


def main():
    """
    Initializes the PapAIrus chat with doc module.

    Loads the settings, creates a RepoAssistant object, extracts data from the project repository, creates a vector store, and launches the Gradio interface.
    """
    logger.info("Initializing the PapAIrus chat with doc module.")

    # Load settings
    setting = SettingsManager.get_setting()
    chat_settings = _select_repo_chat_settings(setting.chat_completion)

    db_path = (
        setting.project.target_repo / setting.project.hierarchy_name / "project_hierarchy.json"
    )

    # Initialize RepoAssistant
    assistant = RepoAssistant(chat_settings, db_path)

    # Extract data
    md_contents, meta_data = assistant.json_data.extract_data()

    # Create vector store and measure runtime
    logger.info("Starting vector store creation...")
    start_time = time.time()
    assistant.vector_store_manager.create_vector_store(md_contents, meta_data)
    elapsed_time = time.time() - start_time
    logger.info(f"Vector store created successfully in {elapsed_time:.2f} seconds.")

    # Launch Gradio interface
    GradioInterface(assistant.respond)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
