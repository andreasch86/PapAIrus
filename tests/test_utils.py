from pathlib import Path

from repo_agent.utils.gitignore_checker import GitignoreChecker
from repo_agent.change_detector import ChangeDetector


def test_gitignore_checker(tmp_path):
    target = tmp_path / "project"
    target.mkdir()
    (target / "include.py").write_text("a=1\n")
    (target / "ignore.py").write_text("b=2\n")
    gitignore = target / ".gitignore"
    gitignore.write_text("ignore.py\n")

    checker = GitignoreChecker(str(target), str(gitignore))
    files = checker.check_files_and_folders()
    assert "include.py" in files
    assert "ignore.py" not in files


def test_gitignore_checker_uses_default_gitignore(tmp_path):
    target = tmp_path / "project"
    target.mkdir()
    (target / "keep.py").write_text("print('ok')\n")

    checker = GitignoreChecker(str(target), str(target / "missing.gitignore"))
    files = checker.check_files_and_folders()
    assert "keep.py" in files


def test_gitignore_checker_skips_folders(tmp_path):
    target = tmp_path / "project"
    target.mkdir()
    ignored_dir = target / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "skip.py").write_text("print('skip')\n")
    (target / "keep.py").write_text("print('keep')\n")
    gitignore = target / ".gitignore"
    gitignore.write_text("ignored/\n")

    checker = GitignoreChecker(str(target), str(gitignore))
    files = checker.check_files_and_folders()
    assert "keep.py" in files
    assert "ignored/skip.py" not in files


def test_is_ignored_handles_trailing_slash():
    assert GitignoreChecker._is_ignored("dir", ["dir/"], is_dir=True)


def test_parse_diffs_and_structure_matching(temp_repo):
    detector = ChangeDetector(temp_repo.working_tree_dir)
    diffs = ["@@ -1,2 +1,3 @@", "+new line", " unchanged", "-old line"]
    parsed = detector.parse_diffs(diffs)
    assert parsed["added"] and parsed["removed"]

    structures = [("func", "sample", 1, 5, None)]
    changes = detector.identify_changes_in_structure(parsed, structures)
    assert changes["added"] or changes["removed"]
