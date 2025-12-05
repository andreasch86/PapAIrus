"""Utilities for generating Google-style docstrings across a repository."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass
class ParameterDoc:
    """Represents documentation metadata for a function parameter."""

    name: str
    annotation: str
    description: str
    prefix: str = ""

    @property
    def display_name(self) -> str:
        """Return the parameter name including any prefix (e.g. "*" or "**")."""

        return f"{self.prefix}{self.name}"


@dataclass
class ReturnDoc:
    """Represents documentation metadata for a function return value."""

    annotation: str
    description: str


class DocstringGenerator:
    """Generate or update docstrings for Python modules in a repository."""

    def __init__(self, root: Path, excluded_directories: Optional[Sequence[str]] = None) -> None:
        self.root = Path(root)
        self.excluded_directories = set(
            excluded_directories
            or {".git", "env", "venv", ".venv", "__pycache__", "node_modules"}
        )

    def run(self, dry_run: bool = False) -> List[Path]:
        """Generate docstrings for all Python files under ``root``.

        Args:
            dry_run: When ``True``, report files that would change without writing
                them.

        Returns:
            A list of file paths that require or received docstring updates.
        """

        updated_files: List[Path] = []
        for file_path in self._python_files():
            if self._update_file(file_path, dry_run=dry_run):
                updated_files.append(file_path)
        return updated_files

    def _python_files(self) -> Iterable[Path]:
        """Yield Python files inside the target tree while skipping exclusions."""

        for path in self.root.rglob("*.py"):
            if any(part in self.excluded_directories for part in path.parts):
                continue
            yield path

    def _update_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Update a single file with any missing or incomplete docstrings."""

        source_lines = file_path.read_text().splitlines(keepends=True)
        try:
            tree = ast.parse("".join(source_lines))
        except SyntaxError:
            return False

        edits: List[tuple[int, int, List[str], bool]] = []
        self._collect_docstring_edits(tree, source_lines, edits)

        if not edits:
            return False

        new_source = self._apply_edits(source_lines, edits)
        if not dry_run:
            file_path.write_text("".join(new_source))
        return True

    def _collect_docstring_edits(
        self, node: ast.AST, source_lines: Sequence[str], edits: List[tuple[int, int, List[str], bool]]
    ) -> None:
        """Recursively collect docstring edits for relevant nodes."""

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._maybe_add_docstring(child, source_lines, edits)
            self._collect_docstring_edits(child, source_lines, edits)

    def _maybe_add_docstring(
        self, node: ast.AST, source_lines: Sequence[str], edits: List[tuple[int, int, List[str], bool]]
    ) -> None:
        existing_docstring = ast.get_docstring(node, clean=False)
        has_docstring = existing_docstring is not None
        body_indent = " " * (getattr(node, "col_offset", 0) + 4)

        docstring_lines = self._build_docstring_if_needed(node, existing_docstring, body_indent)
        if docstring_lines is None:
            return

        if has_docstring:
            doc_expr = node.body[0]
            start = doc_expr.lineno - 1
            end = doc_expr.end_lineno - 1  # type: ignore[attr-defined]
            edits.append((start, end, docstring_lines, True))
        else:
            first_body_line = node.body[0].lineno - 1 if node.body else node.lineno  # type: ignore[attr-defined]
            edits.append((first_body_line, first_body_line, docstring_lines, False))

    def _build_docstring_if_needed(
        self, node: ast.AST, existing_docstring: Optional[str], body_indent: str
    ) -> Optional[List[str]]:
        if isinstance(node, ast.ClassDef):
            if existing_docstring:
                return None
            summary = self._summarize_name(node.name)
            return self._format_docstring(summary, [], None, body_indent)

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None

        parameters = self._extract_parameters(node)
        return_info = self._extract_returns(node)
        needs_return = return_info is not None

        if existing_docstring and not self._docstring_incomplete(existing_docstring, parameters, needs_return):
            return None

        summary = self._existing_summary(existing_docstring) or self._summarize_name(node.name)
        return self._format_docstring(summary, parameters, return_info, body_indent)

    def _docstring_incomplete(self, docstring: str, parameters: Sequence[ParameterDoc], needs_return: bool) -> bool:
        doc_lower = docstring.lower()
        missing_params = [param for param in parameters if param.name.lower() not in doc_lower]
        missing_return = needs_return and "returns:" not in doc_lower
        return bool(missing_params or missing_return)

    def _existing_summary(self, docstring: Optional[str]) -> Optional[str]:
        if not docstring:
            return None
        stripped = docstring.strip().splitlines()
        return stripped[0].strip() if stripped else None

    def _extract_parameters(self, node: ast.AST) -> List[ParameterDoc]:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        params: List[ParameterDoc] = []

        def annotation_to_str(annotation: Optional[ast.AST]) -> str:
            if annotation is None:
                return "Any"
            try:
                return ast.unparse(annotation)
            except Exception:
                return "Any"

        positional_args = list(node.args.posonlyargs) + list(node.args.args)
        for arg in positional_args:
            if arg.arg in {"self", "cls"}:
                continue
            params.append(
                ParameterDoc(
                    name=arg.arg,
                    annotation=annotation_to_str(arg.annotation),
                    description=self._describe_entity(arg.arg),
                )
            )

        if node.args.vararg:
            arg = node.args.vararg
            params.append(
                ParameterDoc(
                    name=arg.arg,
                    annotation=annotation_to_str(arg.annotation),
                    description=self._describe_entity(arg.arg),
                    prefix="*",
                )
            )

        for arg in node.args.kwonlyargs:
            params.append(
                ParameterDoc(
                    name=arg.arg,
                    annotation=annotation_to_str(arg.annotation),
                    description=self._describe_entity(arg.arg),
                )
            )

        if node.args.kwarg:
            arg = node.args.kwarg
            params.append(
                ParameterDoc(
                    name=arg.arg,
                    annotation=annotation_to_str(arg.annotation),
                    description=self._describe_entity(arg.arg),
                    prefix="**",
                )
            )

        return params

    def _extract_returns(self, node: ast.AST) -> Optional[ReturnDoc]:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

        returns_value = any(
            isinstance(child, ast.Return) and child.value is not None
            for child in ast.walk(node)
        ) or any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node))

        if not returns_value:
            return None

        annotation = "Any"
        if node.returns is not None:
            try:
                annotation = ast.unparse(node.returns)
            except Exception:
                annotation = "Any"

        return ReturnDoc(annotation=annotation, description="Description of return value.")

    def _format_docstring(
        self,
        summary: str,
        parameters: Sequence[ParameterDoc],
        return_info: Optional[ReturnDoc],
        body_indent: str,
    ) -> List[str]:
        """Format a Google-style docstring with the provided pieces."""

        clean_summary = summary.rstrip(".") + "."
        lines = [f'{body_indent}"""', f"{body_indent}{clean_summary}"]

        if parameters:
            lines.append("")
            lines.append(f"{body_indent}Args:")
            for param in parameters:
                lines.append(
                    f"{body_indent}{param.display_name} ({param.annotation}): {param.description}"
                )

        if return_info:
            lines.append("")
            lines.append(f"{body_indent}Returns:")
            lines.append(
                f"{body_indent}{return_info.annotation}: {return_info.description}"
            )

        lines.append(f'{body_indent}"""')
        return [f"{line}\n" for line in lines]

    def _summarize_name(self, name: str) -> str:
        """Generate a short summary sentence from an identifier name."""

        words = name.replace("_", " ").strip()
        if not words:
            return "Describe the object."
        formatted = words[0].upper() + words[1:]
        return f"{formatted}."

    def _describe_entity(self, name: str) -> str:
        words = name.replace("_", " ").strip()
        if not words:
            return "Description."
        return f"Description of {words}."

    def _apply_edits(
        self, source_lines: Sequence[str], edits: List[tuple[int, int, List[str], bool]]
    ) -> List[str]:
        """Apply collected edits to the source lines in reverse order."""

        new_lines = list(source_lines)
        for start, end, replacement, is_replacement in sorted(edits, key=lambda item: item[0], reverse=True):
            if is_replacement:
                new_lines[start : end + 1] = replacement
            else:
                new_lines[start:start] = replacement
        return new_lines
