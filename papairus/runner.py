import json
import os
import subprocess
import threading
import time
from functools import partial

from colorama import Fore, Style

from papairus.change_detector import ChangeDetector
from papairus.chat_engine import ChatEngine
from papairus.context_gatherer import ContextGatherer
from papairus.doc_meta_info import DocItem, DocItemStatus, MetaInfo, need_to_generate
from papairus.documentation_updater import DocumentationUpdater
from papairus.file_handler import FileHandler
from papairus.log import logger
from papairus.markdown_generator import MarkdownGenerator
from papairus.multi_task_dispatch import worker
from papairus.project_manager import ProjectManager
from papairus.settings import SettingsManager
from papairus.utils.meta_info_utils import delete_fake_files, make_fake_files


class Runner:
    def __init__(self):
        self.setting = SettingsManager.get_setting()
        self.absolute_project_hierarchy_path = (
            self.setting.project.target_repo / self.setting.project.hierarchy_name
        )

        self.project_manager = ProjectManager(
            repo_path=self.setting.project.target_repo,
            project_hierarchy=self.setting.project.hierarchy_name,
        )
        self.change_detector = ChangeDetector(repo_path=self.setting.project.target_repo)

        self.context_gatherer = ContextGatherer(self.setting)
        global_context = self.context_gatherer.gather()
        self.chat_engine = ChatEngine(
            project_manager=self.project_manager, global_context=global_context
        )

        if not self.absolute_project_hierarchy_path.exists():
            file_path_reflections, jump_files = make_fake_files()
            self.meta_info = MetaInfo.init_meta_info(file_path_reflections, jump_files)
            self.meta_info.checkpoint(target_dir_path=self.absolute_project_hierarchy_path)
        else:
            # Load existing project hierarchy from checkpoint
            self.meta_info = MetaInfo.from_checkpoint_path(self.absolute_project_hierarchy_path)

        self.meta_info.checkpoint(target_dir_path=self.absolute_project_hierarchy_path)
        self.runner_lock = threading.Lock()

        self.markdown_generator = MarkdownGenerator(self.setting, self.meta_info, self.runner_lock)
        self.documentation_updater = DocumentationUpdater(
            self.project_manager, self.chat_engine, self.setting
        )

    def generate_doc_for_a_single_item(self, doc_item: DocItem):
        """Generate documentation for a single item."""
        try:
            if not need_to_generate(doc_item, self.setting.project.ignore_list):
                print(f"Content ignored/Document generated, skipping: {doc_item.get_full_name()}")
            else:
                print(
                    f" -- Generating document  {Fore.LIGHTYELLOW_EX}{doc_item.item_type.name}: {doc_item.get_full_name()}{Style.RESET_ALL}"
                )
                response_message = self.chat_engine.generate_doc(
                    doc_item=doc_item,
                )
                doc_item.md_content.append(response_message)  # type: ignore
                doc_item.item_status = DocItemStatus.doc_up_to_date
                self.meta_info.checkpoint(target_dir_path=self.absolute_project_hierarchy_path)
        except Exception:
            logger.exception(
                f"Document generation failed after multiple attempts, skipping: {doc_item.get_full_name()}"
            )
            doc_item.item_status = DocItemStatus.doc_has_not_been_generated

    def first_generate(self):
        """
        First generation of documentation.
        """
        logger.info("Starting to generate documentation")
        check_task_available_func = partial(
            need_to_generate, ignore_list=self.setting.project.ignore_list
        )
        task_manager = self.meta_info.get_topology(check_task_available_func)
        before_task_len = len(task_manager.task_dict)

        if not self.meta_info.in_generation_process:
            self.meta_info.in_generation_process = True
            logger.info("Init a new task-list")
        else:
            logger.info("Load from an existing task-list")
        self.meta_info.print_task_list(task_manager.task_dict)

        try:
            # Create threads for concurrent generation
            threads = [
                threading.Thread(
                    target=worker,
                    args=(
                        task_manager,
                        process_id,
                        self.generate_doc_for_a_single_item,
                    ),
                )
                for process_id in range(self.setting.project.max_thread_count)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Refresh markdown files
            self.markdown_generator.refresh()

            # Update document version to current commit
            self.meta_info.document_version = self.change_detector.repo.head.commit.hexsha
            self.meta_info.in_generation_process = False
            self.meta_info.checkpoint(target_dir_path=self.absolute_project_hierarchy_path)
            logger.info(
                f"Successfully generated {before_task_len - len(task_manager.task_dict)} documents."
            )

        except BaseException as e:
            logger.error(
                f"An error occurred: {e}. {before_task_len - len(task_manager.task_dict)} docs are generated at this time"
            )

    def git_commit(self, commit_message):
        try:
            subprocess.check_call(
                ["git", "commit", "--no-verify", "-m", commit_message],
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while trying to commit {str(e)}")

    def run(self):
        """
        Runs the document update process.

        This method detects the changed Python files, processes each file, and updates the documents accordingly.

        Returns:
            None
        """

        if self.meta_info.document_version == "":
            # First generation process
            self.first_generate()
            self.meta_info.checkpoint(
                target_dir_path=self.absolute_project_hierarchy_path,
                flash_reference_relation=True,
            )
            return

        if not self.meta_info.in_generation_process:
            logger.info("Starting to detect changes.")

            """
            1. Load previous project hierarchy.
            2. Merge hierarchy:
            - Update: Update doc
            - Add/Remove: Update doc
            - Change: Update obj-doc

            Merge into new meta info:
            1. Merge meta info
            2. Update objects
            3. Checkpoint
            """
            file_path_reflections, jump_files = make_fake_files()
            new_meta_info = MetaInfo.init_meta_info(file_path_reflections, jump_files)
            new_meta_info.load_doc_from_older_meta(self.meta_info)

            self.meta_info = new_meta_info
            self.meta_info.in_generation_process = True

        check_task_available_func = partial(
            need_to_generate, ignore_list=self.setting.project.ignore_list
        )

        task_manager = self.meta_info.get_task_manager(
            self.meta_info.target_repo_hierarchical_tree,
            task_available_func=check_task_available_func,
        )

        for item_name, item_type in self.meta_info.deleted_items_from_older_meta:
            print(
                f"{Fore.LIGHTMAGENTA_EX}[Dir/File/Obj Delete Dected]: {Style.RESET_ALL} {item_type} {item_name}"
            )
        self.meta_info.print_task_list(task_manager.task_dict)
        if task_manager.all_success:
            logger.info("No tasks in the queue, all documents are completed and up to date.")

        threads = [
            threading.Thread(
                target=worker,
                args=(task_manager, process_id, self.generate_doc_for_a_single_item),
            )
            for process_id in range(self.setting.project.max_thread_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.meta_info.in_generation_process = False
        self.meta_info.document_version = self.change_detector.repo.head.commit.hexsha

        self.meta_info.checkpoint(
            target_dir_path=self.absolute_project_hierarchy_path,
            flash_reference_relation=True,
        )
        logger.info("Doc has been forwarded to the latest version")

        self.markdown_generator.refresh()
        delete_fake_files()

        logger.info("Starting to git-add DocMetaInfo and newly generated Docs")
        time.sleep(1)

        # Run git add for unstaged files
        git_add_result = self.change_detector.add_unstaged_files()

        if len(git_add_result) > 0:
            logger.info(f"Added {[file for file in git_add_result]} to the staging area.")

    def process_file_changes(self, repo_path, file_path, is_new_file):
        """
        This function is called in the loop of detected changed files. Its purpose is to process changed files according to the absolute file path, including new files and existing files.
        Among them, changes_in_pyfile is a dictionary that contains information about the changed structures. An example format is: {'added': {'add_context_stack', '__init__'}, 'removed': set()}

        Args:
            repo_path (str): The path to the repository.
            file_path (str): The relative path to the file.
            is_new_file (bool): Indicates whether the file is new or not.

        Returns:
            None
        """

        file_handler = FileHandler(repo_path=repo_path, file_path=file_path)
        # Read source code
        source_code = file_handler.read_file()
        changed_lines = self.change_detector.parse_diffs(
            self.change_detector.get_file_diff(file_path, is_new_file)
        )
        changes_in_pyfile = self.change_detector.identify_changes_in_structure(
            changed_lines, file_handler.get_functions_and_classes(source_code)
        )
        logger.info(f"Changes detected:\n{changes_in_pyfile}")

        # Update project hierarchy
        with open(self.project_manager.project_hierarchy, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if file_handler.file_path in json_data:
            # Update existing item
            json_data[file_handler.file_path] = self.documentation_updater.update_existing_item(
                json_data[file_handler.file_path], file_handler, changes_in_pyfile
            )
            # Write updated json
            with open(self.project_manager.project_hierarchy, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            logger.info(f"Updated json for {file_handler.file_path}.")

            # Convert to markdown
            markdown = file_handler.convert_to_markdown_file(file_path=file_handler.file_path)
            # Write markdown file
            file_handler.write_file(
                os.path.join(
                    self.setting.project.markdown_docs_name,
                    file_handler.file_path.replace(".py", ".md"),
                ),
                markdown,
            )
            logger.info(f"Updated markdown for {file_handler.file_path}.")

        else:
            self.documentation_updater.add_new_item(file_handler, json_data)

        # Run git add
        git_add_result = self.change_detector.add_unstaged_files()

        if len(git_add_result) > 0:
            logger.info(f"Added {[file for file in git_add_result]} to staging area.")


if __name__ == "__main__":
    runner = Runner()

    runner.run()

    logger.info("Done.")
