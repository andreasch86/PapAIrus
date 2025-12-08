import shutil
import time
from pathlib import Path
from tqdm import tqdm
from papairus.log import logger

class MarkdownGenerator:
    def __init__(self, setting, meta_info, lock):
        self.setting = setting
        self.meta_info = meta_info
        self.lock = lock

    def refresh(self):
        with self.lock:
            # Delete existing markdown folder
            markdown_folder = (
                Path(self.setting.project.target_repo) / self.setting.project.markdown_docs_name
            )

            if markdown_folder.exists():
                logger.debug(f"Deleting existing contents of {markdown_folder}")
                shutil.rmtree(markdown_folder)
            markdown_folder.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created markdown folder at {markdown_folder}")

        # Process all files
        file_item_list = self.meta_info.get_all_files()
        logger.debug(f"Found {len(file_item_list)} files to process.")

        for file_item in tqdm(file_item_list):
            def recursive_check(doc_item) -> bool:
                if doc_item.md_content:
                    return True
                for child in doc_item.children.values():
                    if recursive_check(child):
                        return True
                return False

            if not recursive_check(file_item):
                logger.debug(
                    f"No documentation content for: {file_item.get_full_name()}, skipping."
                )
                continue

            # Generate markdown content
            markdown = ""
            for child in file_item.children.values():
                markdown += self.to_markdown(child, 2)

            if not markdown:
                logger.warning(f"No markdown content generated for: {file_item.get_full_name()}")
                continue

            # Determine file path
            file_path = Path(
                self.setting.project.markdown_docs_name
            ) / file_item.get_file_name().replace(".py", ".md")
            abs_file_path = self.setting.project.target_repo / file_path
            logger.debug(f"Writing markdown to: {abs_file_path}")

            # Ensure parent directory exists
            abs_file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {abs_file_path.parent}")

            # Write file with retry logic
            with self.lock:
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        with open(abs_file_path, "w", encoding="utf-8") as file:
                            file.write(markdown)
                        logger.debug(f"Successfully wrote to {abs_file_path}")
                        break
                    except IOError as e:
                        logger.error(
                            f"Failed to write {abs_file_path} on attempt {attempt + 1}: {e}"
                        )
                        time.sleep(1)

        logger.info(
            f"Markdown documents have been refreshed at {self.setting.project.markdown_docs_name}"
        )

    def to_markdown(self, item, now_level: int) -> str:
        """Convert item to markdown."""
        markdown_content = "#" * now_level + f" {item.item_type.to_str()} {item.obj_name}"
        if "params" in item.content.keys() and item.content["params"]:
            markdown_content += f"({', '.join(item.content['params'])})"
        markdown_content += "\n"
        if item.md_content:
            markdown_content += f"{item.md_content[-1]}\n"
        else:
            markdown_content += "Doc is waiting to be generated...\n"
        for child in item.children.values():
            markdown_content += self.to_markdown(child, now_level + 1)
            markdown_content += "***\n"
        return markdown_content
