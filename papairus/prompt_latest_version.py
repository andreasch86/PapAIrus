from papairus.llm.backends.base import ChatMessage

doc_generation_instruction = (
    "You are a Senior Technical Writer. You are documenting the file `{file_path}` for the project `{project_name}`.\n\n"
    "**Context:**\n"
    "The project entry point is: {entry_point_summary}\n"
    "Users typically interact with this code by: {usage_context_from_tests}\n\n"
    "**Your Task:**\n"
    "Write a Markdown guide for `{code_name}`.\n"
    "1. Start with a high-level summary of *why* a user would import this.\n"
    "2. Create a 'Usage' section. Look at the provided test code: `{test_content}`. Adapt this into a clean, copy-pasteable example for the user.\n"
    "3. API Reference: List parameters only if they are not obvious.\n"
    "4. Do not include internal jargon. Do not use 'ClassDef' in titles. Use natural language headings.\n\n"
    "The content of the code is as follows:\n"
    "{code_content}\n"
)

documentation_guideline = (
    "Write for readers who want a holistic understanding of the repository. Be factual, avoid speculation, and weave in insights from both code and docstrings. "
    "Keep the tone clear and supportive, and focus on accurately reflecting repository behaviour in {language}."
)


def build_repo_documentation_messages(**kwargs) -> list[ChatMessage]:
    """
    Assemble context-aware messages for documentation/chat mode.

    Args:
        **kwargs: Keyword arguments passed to the `doc_generation_instruction` and `documentation_guideline` functions.

    Returns:
        A list of `ChatMessage` objects containing the system and user messages.
    """

    system_message = doc_generation_instruction.format(**kwargs)
    user_message = documentation_guideline.format(language=kwargs.get("language", "English"))
    return [
        ChatMessage(content=system_message, role="system"),
        ChatMessage(content=user_message, role="user"),
    ]


def build_docstring_messages(code_snippet: str, *, style: str = "google") -> list[ChatMessage]:
    """
    Build a strict docstring few-shot prompt.

    Args:
        code_snippet: The code snippet to generate a docstring for.
        style: The style of docstring to generate. Defaults to "google".

    Returns:
        A list of `ChatMessage` objects containing the prompt.
    """

    header = (
        "You are a Python docstring generator. Produce ONLY a docstring in Google style with Args/Returns/Raises"
        " matching the provided code. Do not add commentary or prefixes."
    )
    instructions = f"Format: {style} docstring."
    prompt = "\n\n".join([header, instructions, "Code:", code_snippet])
    return [ChatMessage(role="system", content=prompt)]
