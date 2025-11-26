from papairus.doc_meta_info import (
    DocItem,
    DocItemStatus,
    DocItemType,
    need_to_generate,
)


def build_tree():
    root = DocItem(item_type=DocItemType._repo, obj_name="root")
    directory = DocItem(item_type=DocItemType._dir, obj_name="pkg", father=root)
    file_item = DocItem(item_type=DocItemType._file, obj_name="module.py", father=directory)
    function = DocItem(
        item_type=DocItemType._function,
        obj_name="func",
        father=file_item,
        item_status=DocItemStatus.doc_has_not_been_generated,
    )
    root.children["pkg"] = directory
    directory.children["module.py"] = file_item
    file_item.children["func"] = function
    root.parse_tree_path([])
    directory.parse_tree_path(root.tree_path)
    file_item.parse_tree_path(directory.tree_path)
    function.parse_tree_path(file_item.tree_path)
    return root, directory, file_item, function


def test_need_to_generate_respects_status_and_ignore():
    _, _, file_item, function = build_tree()
    assert need_to_generate(function)
    assert not need_to_generate(file_item)
    assert not need_to_generate(function, ignore_list=["pkg/module.py"])


def test_tree_helpers_compute_depth_and_paths():
    root, directory, file_item, function = build_tree()
    assert function.get_file_name().endswith("module.py")
    assert function.check_depth() == 0
    assert root.check_depth() == 3
    travel = root.get_travel_list()
    assert root in travel and function in travel
    assert DocItem.has_ans_relation(file_item, function) == file_item
