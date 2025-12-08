import json
import os
from concurrent.futures import ThreadPoolExecutor

from colorama import Fore, Style

from papairus.doc_meta_info import DocItem
from papairus.log import logger


class DocumentationUpdater:
    def __init__(self, project_manager, chat_engine, setting):
        self.project_manager = project_manager
        self.chat_engine = chat_engine
        self.setting = setting

    def add_new_item(self, file_handler, json_data):
        """
        Add new projects to the JSON file and generate corresponding documentation.

        Args:
            file_handler (FileHandler): The file handler object for reading and writing files.
            json_data (dict): The JSON data storing the project structure information.

        Returns:
            None
        """
        file_dict = {}
        # Iterate over functions and classes
        for (
            structure_type,
            name,
            start_line,
            end_line,
            parent,
            params,
        ) in file_handler.get_functions_and_classes(file_handler.read_file()):
            code_info = file_handler.get_obj_code_info(
                structure_type, name, start_line, end_line, parent, params
            )
            doc_item = DocItem(obj_name=name, content=code_info)
            # Manually set item_type if possible, but DocItem defaults to _class_function
            md_content = self.chat_engine.generate_doc(doc_item)
            code_info["md_content"] = md_content
            # Update file dict
            file_dict[name] = code_info

        json_data[file_handler.file_path] = file_dict
        # Write json data
        with open(self.project_manager.project_hierarchy, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        logger.info(
            f"The structural information of the newly added file {file_handler.file_path} has been written into a JSON file."
        )
        # Convert to markdown
        markdown = file_handler.convert_to_markdown_file(file_path=file_handler.file_path)
        # Write markdown file
        file_handler.write_file(
            os.path.join(
                self.project_manager.repo_path,
                self.setting.project.markdown_docs_name,
                file_handler.file_path.replace(".py", ".md"),
            ),
            markdown,
        )
        logger.info(f"Generated markdown for {file_handler.file_path}.")

    def update_existing_item(self, file_dict, file_handler, changes_in_pyfile):
        """
        Update existing projects.

        Args:
            file_dict (dict): A dictionary containing file structure information.
            file_handler (FileHandler): The file handler object.
            changes_in_pyfile (dict): A dictionary containing information about the objects that have changed in the file.

        Returns:
            dict: The updated file structure information dictionary.
        """
        new_obj, del_obj = self.get_new_objects(file_handler)

        # Remove deleted objects
        for obj_name in del_obj:
            if obj_name in file_dict:
                del file_dict[obj_name]
                logger.info(f"Removed {obj_name}.")

        referencer_list = []

        # Get current file structure
        current_objects = file_handler.generate_file_structure(file_handler.file_path)

        current_info_dict = {obj["name"]: obj for obj in current_objects.values()}

        # Update or add objects
        for current_obj_name, current_obj_info in current_info_dict.items():
            if current_obj_name in file_dict:
                # Update existing object info
                file_dict[current_obj_name]["type"] = current_obj_info["type"]
                file_dict[current_obj_name]["code_start_line"] = current_obj_info["code_start_line"]
                file_dict[current_obj_name]["code_end_line"] = current_obj_info["code_end_line"]
                file_dict[current_obj_name]["parent"] = current_obj_info["parent"]
                file_dict[current_obj_name]["name_column"] = current_obj_info["name_column"]
            else:
                # Add new object info
                file_dict[current_obj_name] = current_obj_info

        # Process added changes
        for obj_name, _ in changes_in_pyfile["added"]:
            for current_object in current_objects.values():
                if obj_name == current_object["name"]:
                    # Create referencer object
                    referencer_obj = {
                        "obj_name": obj_name,
                        "obj_referencer_list": self.project_manager.find_all_referencer(
                            variable_name=current_object["name"],
                            file_path=file_handler.file_path,
                            line_number=current_object["code_start_line"],
                            column_number=current_object["name_column"],
                        ),
                    }
                    referencer_list.append(referencer_obj)

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit updates
            futures = []
            for changed_obj in changes_in_pyfile["added"]:
                for ref_obj in referencer_list:
                    if changed_obj[0] == ref_obj["obj_name"]:
                        future = executor.submit(
                            self.update_object,
                            file_dict,
                            file_handler,
                            changed_obj[0],
                            ref_obj["obj_referencer_list"],
                        )
                        print(
                            f"Updating {Fore.CYAN}{file_handler.file_path}{Style.RESET_ALL} -> {Fore.CYAN}{changed_obj[0]}{Style.RESET_ALL}."
                        )
                        futures.append(future)

            for future in futures:
                future.result()

        return file_dict

    def update_object(self, file_dict, file_handler, obj_name, obj_referencer_list):
        """
        Generate documentation content and update corresponding field information of the object.

        Args:
            file_dict (dict): A dictionary containing old object information.
            file_handler: The file handler.
            obj_name (str): The object name.
            obj_referencer_list (list): The list of object referencers.

        Returns:
            None
        """
        if obj_name in file_dict:
            obj = file_dict[obj_name]
            doc_item = DocItem(obj_name=obj_name, content=obj)
            response_message = self.chat_engine.generate_doc(doc_item)
            obj["md_content"] = response_message

    def get_new_objects(self, file_handler):
        """
        The function gets the added and deleted objects by comparing the current version and the previous version of the .py file.

        Args:
            file_handler (FileHandler): The file handler object.

        Returns:
            tuple: A tuple containing the added and deleted objects, in the format (new_obj, del_obj)

        Output example:
            new_obj: ['add_context_stack', '__init__']
            del_obj: []
        """
        current_version, previous_version = file_handler.get_modified_file_versions()
        parse_current_py = file_handler.get_functions_and_classes(current_version)
        parse_previous_py = (
            file_handler.get_functions_and_classes(previous_version) if previous_version else []
        )

        current_obj = {f[1] for f in parse_current_py}
        previous_obj = {f[1] for f in parse_previous_py}

        new_obj = list(current_obj - previous_obj)
        del_obj = list(previous_obj - current_obj)
        return new_obj, del_obj
