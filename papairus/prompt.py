from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

doc_generation_instruction = (
    "You are an AI documentation assistant. Read the code and its Google-style docstrings to write friendly, accurate "
    "feature documentation. Avoid jargon where possible and include short examples only when they meaningfully clarify behaviour.\n\n"
    "Currently, you are in a project{project_structure_prefix}\n"
    "{project_structure}\n\n"
    "The path of the document you need to generate in this project is {file_path}.\n"
    'Now you need to generate a document for a {code_type_tell}, whose name is "{code_name}".\n\n'
    "The content of the code is as follows:\n"
    "{code_content}\n\n"
    "{reference_letter}\n"
    "{referencer_content}\n\n"
    "Use the docstrings and source code to explain what this item does {combine_ref_situation}. Summarise inputs, outputs, and relationships in clear, welcoming language.\n\n"
    "The standard format is as follows (omit sections with no relevant details):\n\n"
    "**{code_name}**: Friendly one-sentence purpose.\n"
    "**{parameters_or_attribute}**: The {parameters_or_attribute} of this {code_type_tell}.\n"
    "· parameter1: what it represents\n"
    "· ...\n"
    "**Code Description**: A concise, confident explanation of how it works{has_relationship}.\n"
    "**Note**: Helpful usage notes or caveats.\n"
    "{have_return_tell}\n\n"
    "Please note:\n"
    "- Do not use Markdown headings or dividers.\n"
    "- Prefer approachable wording while staying technically correct.\n"
    "- Only include examples when they directly improve clarity.\n"
)

documentation_guideline = (
    "Write for readers who want a holistic understanding of the repository. Be factual, avoid speculation, and weave in insights from both code and docstrings. "
    "Keep the tone clear and supportive, and focus on accurately reflecting repository behaviour in {language}."
)


message_templates = [
    ChatMessage(content=doc_generation_instruction, role=MessageRole.SYSTEM),
    ChatMessage(
        content=documentation_guideline,
        role=MessageRole.USER,
    ),
]

chat_template = ChatPromptTemplate(message_templates=message_templates)
