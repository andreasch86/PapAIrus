from papairus.doc_meta_info import DocItem
from papairus.llm_provider import build_llm
from papairus.log import logger
from papairus.prompt import build_repo_documentation_messages
from papairus.settings import SettingsManager


class ChatEngine:
    """
    ChatEngine is used to generate the doc of functions or classes.
    """

    def __init__(self, project_manager, global_context=None):
        setting = SettingsManager.get_setting()

        self.llm = build_llm(setting.chat_completion)
        self.project_manager = project_manager
        self.global_context = global_context or {}

    def build_prompt(self, doc_item: DocItem):
        """Builds and returns the system and user prompts based on the DocItem."""
        setting = SettingsManager.get_setting()

        code_info = doc_item.content
        code_name = code_info["name"]
        code_content = code_info["code_content"]
        file_path = doc_item.get_full_name()

        # Find test content
        target_name = doc_item.get_file_name().split("/")[-1].replace(".py", "")
        test_content = "No specific test found."
        tests_map = self.global_context.get("tests_map", {})

        for t_name, t_content in tests_map.items():
            if target_name in t_name:
                test_content = t_content[:2000]  # Limit size
                break

        entry_point = self.global_context.get("entry_point_summary", "No entry point found.")
        existing_docs = self.global_context.get("existing_docs_sample")
        if existing_docs:
            entry_point += f"\n\nExisting Documentation Style Sample:\n{existing_docs}"

        return build_repo_documentation_messages(
            file_path=file_path,
            project_name=self.global_context.get("project_name", "Project"),
            entry_point_summary=entry_point,
            usage_context_from_tests=self.global_context.get(
                "usage_context_from_tests", "No usage context found."
            ),
            code_name=code_name,
            test_content=test_content,
            code_content=code_content,
            language=setting.project.language,
        )

    def generate_doc(self, doc_item: DocItem):
        """Generates documentation for a given DocItem."""
        messages = self.build_prompt(doc_item)

        try:
            response = self.llm.generate_response(messages)
            logger.debug(f"LLM Prompt Tokens: {response.raw.usage.prompt_tokens}")  # type: ignore
            logger.debug(
                f"LLM Completion Tokens: {response.raw.usage.completion_tokens}"  # type: ignore
            )
            logger.debug(
                f"Total LLM Token Count: {response.raw.usage.total_tokens}"  # type: ignore
            )
            return response.message.content
        except Exception as e:
            logger.error(f"Error in llamaindex chat call: {e}")
            raise
