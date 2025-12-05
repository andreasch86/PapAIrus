from llama_index.core.llms.function_calling import FunctionCallingLLM

from papairus.chat_with_repo.json_handler import JsonFileProcessor


class TextAnalysisTool:
    def __init__(self, llm: FunctionCallingLLM, db_path):
        self.jsonsearch = JsonFileProcessor(db_path)
        self.llm = llm
        self.db_path = db_path

    def keyword(self, query):
        prompt = f"Please provide a list of Code keywords according to the following query, please output no more than 3 keywords, Input: {query}, Output:"
        response = self.llm.complete(prompt)
        return response

    def tree(self, query):
        prompt = f"Please analyze the following text and generate a tree structure based on its hierarchy:\n\n{query}"
        response = self.llm.complete(prompt)
        return response

    def format_chat_prompt(self, message, instruction):
        prompt = f"System:{instruction}\nUser: {message}\nAssistant:"
        return prompt

    def queryblock(self, message):
        search_result, md = self.jsonsearch.search_code_contents_by_name(self.db_path, message)
        return search_result, md

    def list_to_markdown(self, search_result):
        markdown_str = ""
        # English，EnglishMarkdownEnglish
        for index, content in enumerate(search_result, start=1):
            # EnglishMarkdownEnglish，English
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
    from papairus.llm.backends.gemini import GeminiBackend

    api_key = "your_api_key"
    log_file = "your_logfile_path"
    llm = GeminiBackend(
        api_key=api_key,
        base_url="https://aiplatform.googleapis.com/v1",
        model="gemini-2.5-flash",
        temperature=0.2,
        timeout=60,
    )
    db_path = "your_database_path"
    test = TextAnalysisTool(llm, db_path)
