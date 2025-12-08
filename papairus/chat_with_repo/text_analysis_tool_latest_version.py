from llama_index.core.llms.function_calling import FunctionCallingLLM

from papairus.chat_with_repo.json_handler import JsonFileProcessor


class TextAnalysisTool:
    """
    A tool for analyzing text and generating code-related information.

    Args:
    llm: An instance of FunctionCallingLLM.
    db_path: The path to the database containing code information.
    """

    def __init__(self, llm: FunctionCallingLLM, db_path):
        self.jsonsearch = JsonFileProcessor(db_path)
        self.llm = llm
        self.db_path = db_path

    def keyword(self, query):
        """
        Generates a list of code keywords based on a given query.

        Args:
        query: The query to generate keywords for.

        Returns:
        A list of code keywords.
        """
        prompt = f"Please provide a list of Code keywords according to the following query, please output no more than 3 keywords, Input: {query}, Output:"
        response = self.llm.complete(prompt)
        return response

    def tree(self, query):
        """
        Generates a tree structure based on the hierarchy of a given text.

        Args:
        query: The text to generate a tree structure for.

        Returns:
        A tree structure.
        """
        prompt = f"Please analyze the following text and generate a tree structure based on its hierarchy:\n\n{query}"
        response = self.llm.complete(prompt)
        return response

    def format_chat_prompt(self, message, instruction):
        """
        Formats a chat prompt with a given message and instruction.

        Args:
        message: The message to include in the prompt.
        instruction: The instruction to include in the prompt.

        Returns:
        A formatted chat prompt.
        """
        prompt = f"System:{instruction}\nUser: {message}\nAssistant:"
        return prompt

    def queryblock(self, message):
        """
        Searches for code blocks in a given message and returns the corresponding code and markdown.

        Args:
        message: The message to search for code blocks in.

        Returns:
        A tuple containing the code and markdown.
        """
        search_result, md = self.jsonsearch.search_code_contents_by_name(self.db_path, message)
        return search_result, md

    def list_to_markdown(self, search_result):
        markdown_str = ""
        for index, content in enumerate(search_result, start=1):
            markdown_str += f"{index}. {content}\n\n"

        return markdown_str

    def nerquery(self, message):
        instrcution = """
Extract the most relevant class or function base on the following instrcution:

The output must strictly be a pure function name or class name, without any additional characters.
For example:
Pure function names: calculateSum, processData
Pure class names: MyClass, DataProcessor
The output function name or class name should be only one.
        """
        query = f"{instrcution}\n\nThe input is shown as bellow:\n{message}\n\nAnd now directly give your Output:"
        response = self.llm.complete(query)
        # logger.debug(f"Input: {message}, Output: {response}")
        return response


if __name__ == "__main__":  # pragma: no cover - manual demonstration helper
    from papairus.llm.backends.local_gemma import LocalGemmaBackend

    llm = LocalGemmaBackend(
        model="codegemma:instruct",
        base_url="http://localhost:11434",
        temperature=0.2,
        request_timeout=60,
    )
    db_path = "your_database_path"
    test = TextAnalysisTool(llm, db_path)
