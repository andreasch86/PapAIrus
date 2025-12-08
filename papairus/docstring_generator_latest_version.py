"""Utilities for generating Google-style docstrings across a repository."""

from __future__ import annotations

import ast
import re
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

from papairus.llm.backends.base import ChatMessage, LLMBackend


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

    def __init__(
        self,
        root: Path,
        excluded_directories: Optional[Sequence[str]] = None,
        backend: str = "ast",
        llm_client: Optional[Any] = None,
        refresh_existing_llm_docstrings: bool = True,
        force: bool = False,
    ) -> None:
        self.root = Path(root)
        self.excluded_directories = set(
            excluded_directories or {".git", "env", "venv", ".venv", "__pycache__", "node_modules"}
        )
        self.backend = backend.lower()
        if self.backend not in {"ast", "gemini", "gemma", "llm"}:
            raise ValueError("backend must be one of: ast, gemini, gemma, llm")
        self.llm_client = llm_client
        self.refresh_existing_llm_docstrings = refresh_existing_llm_docstrings
        self.force = force

    def run(
        self,
        dry_run: bool = False,
        progress_callback: Optional[Callable[[Path, str], None]] = None,
    ) -> List[Path]:
        """Generate docstrings for all Python files under ``root``.

        Args:
            dry_run: When ``True``, report files that would change without writing
                them.

        Returns:
            A list of file paths that require or received docstring updates.
        """

        updated_files: List[Path] = []
        for file_path in self._python_files():
            if progress_callback:
                progress_callback(file_path, "start")

            changed = self._update_file(file_path, dry_run=dry_run)
            if changed:
                updated_files.append(file_path)

            if progress_callback:
                progress_callback(file_path, "updated" if changed else "skipped")
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
        candidates: List[tuple[ast.AST, str, Optional[str]]] = []

        self._collect_candidates(tree, candidates)

        if not candidates:
            return False

        if self.backend == "ast":
            for node, indent, existing_doc in candidates:
                doc_lines = self._build_ast_docstring(node, existing_doc, indent)
                if doc_lines:
                    self._add_edit(node, existing_doc, doc_lines, edits)
        else:
            self._process_batch(candidates, source_lines, edits)

        if not edits:
            return False

        new_source = self._apply_edits(source_lines, edits)
        if not dry_run:
            file_path.write_text("".join(new_source))
        return True

    def _collect_candidates(self, node: ast.AST, candidates: List):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._check_candidate(child, candidates)
            self._collect_candidates(child, candidates)

    def _check_candidate(self, node, candidates):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            return

        existing_docstring = ast.get_docstring(node, clean=False)
        body_indent = " " * (getattr(node, "col_offset", 0) + 4)

        needs_update = False
        if self.force:
            needs_update = True
        elif self.backend == "ast":
            # For AST, we allow all potentially valid nodes to be candidates
            # _build_ast_docstring will check validity and completeness
            needs_update = True
        else:
            refresh_existing = self.refresh_existing_llm_docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = self._extract_parameters(node)
                return_info = self._extract_returns(node)
                needs_return = return_info is not None
                needs_update = refresh_existing or existing_docstring is None
                if not needs_update:
                    needs_update = self._docstring_incomplete(
                        existing_docstring, parameters, needs_return
                    )
            elif isinstance(node, ast.ClassDef):
                needs_update = refresh_existing or existing_docstring is None

        if needs_update:
            candidates.append((node, body_indent, existing_docstring))

    def _add_edit(self, node, existing_docstring, docstring_lines, edits):
        has_docstring = existing_docstring is not None
        if has_docstring:
            doc_expr = node.body[0]
            start = doc_expr.lineno - 1
            end = doc_expr.end_lineno - 1
            edits.append((start, end, docstring_lines, True))
        else:
            first_body_line = node.body[0].lineno - 1 if node.body else node.lineno
            edits.append((first_body_line, first_body_line, docstring_lines, False))

    def _process_batch_item(self, batch, source_text, edits, edits_lock):
        prompt = self._build_batch_prompt(batch, source_text)

        try:
            response = self._call_llm_raw(prompt)
            results = self._parse_batch_response(response, [c[0].name for c in batch])

            for (node, indent, existing_doc), docstring in zip(batch, results):
                if docstring:
                    # Normalize to remove wrapping quotes if present
                    cleaned = self._normalize_llm_output(docstring)
                    lines = self._indent_docstring(cleaned.splitlines(), indent)
                    with edits_lock:
                        self._add_edit(node, existing_doc, lines, edits)
        except Exception:
            # Log error or silence it (DocstringGenerator shouldn't crash process easily)
            pass

    def _process_batch(self, candidates, source_lines, edits):
        batch_size = 5
        source_text = "".join(source_lines)
        edits_lock = threading.Lock()

        batches = []
        for i in range(0, len(candidates), batch_size):
            batches.append(candidates[i : i + batch_size])

        with ThreadPoolExecutor() as executor:
            # Use map to process in parallel
            list(
                executor.map(
                    lambda b: self._process_batch_item(b, source_text, edits, edits_lock), batches
                )
            )

    def _build_batch_prompt(self, batch, source_text):
        header = (
            "Generate Google-style Python docstrings for the following functions/classes. "
            "Output each docstring inside a block delimited by >>> NAME and <<< NAME. "
            "Example:\n"
            ">>> my_function\n"
            '"""\nDescription...\nArgs:\n...\n"""\n'
            "<<< my_function\n\n"
            "Code snippets:"
        )
        snippets = []
        for node, _, existing in batch:
            seg = ast.get_source_segment(source_text, node) or ""
            snippets.append(f"--- {node.name} ---\n{seg}\n")

        return header + "\n\n" + "\n".join(snippets)

    def _parse_batch_response(self, response, names):
        results = []
        for name in names:
            pattern = re.compile(
                rf">>> {re.escape(name)}\s*(.*?)\s*<<< {re.escape(name)}", re.DOTALL
            )
            match = pattern.search(response)
            if match:
                results.append(match.group(1).strip())
            else:
                results.append(None)
        return results

    def _call_llm_raw(self, prompt: str) -> str:
        if self.llm_client is None:
            raise ValueError("LLM backend requires an llm_client instance")

        if isinstance(self.llm_client, LLMBackend):
            messages = [ChatMessage(role="user", content=prompt)]
            response = self.llm_client.generate_response(messages)
            return str(response.message.content)

        if callable(self.llm_client):
            return str(self.llm_client(prompt))

        chat_method = getattr(self.llm_client, "chat", None)
        if callable(chat_method):
            messages = [ChatMessage(role="user", content=prompt)]
            response = chat_method(messages)
            message = getattr(response, "message", response)
            return str(getattr(message, "content", ""))

        raise ValueError("llm_client must be callable or expose a chat(messages) method")

    def _build_ast_docstring(
        self, node: ast.AST, existing_docstring: Optional[str], body_indent: str
    ) -> Optional[List[str]]:
        """
        Generates a Google-style Python docstring for a given AST node.

        Args:
            self: The `DocstringGenerator` instance.
            node: The AST node to generate a docstring for.
            existing_docstring: The existing docstring for the node.
            body_indent: The indentation of the body of the docstring.

        Returns:
            A list of lines for the docstring, or `None` if no docstring should be generated.
        """
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

        if existing_docstring and not self._docstring_incomplete(
            existing_docstring, parameters, needs_return
        ):
            return None

        summary = self._existing_summary(existing_docstring) or self._summarize_name(node.name)
        return self._format_docstring(summary, parameters, return_info, body_indent)

    def _docstring_incomplete(
        self, docstring: str, parameters: Sequence[ParameterDoc], needs_return: bool
    ) -> bool:
        """
        Checks if a docstring is incomplete.

        Args:
            self: The `DocstringGenerator` instance.
            docstring: The docstring to check.
            parameters: The parameters of the function or method.
            needs_return: Whether the function or method returns a value.

        Returns:
            True if the docstring is incomplete, False otherwise.
        """
        doc_lower = docstring.lower()
        missing_params = [param for param in parameters if param.name.lower() not in doc_lower]
        missing_return = needs_return and "returns:" not in doc_lower
        return bool(missing_params or missing_return)

    def _existing_summary(self, docstring: Optional[str]) -> Optional[str]:
        """
        Extracts the summary from an existing docstring.

        Args:
            self: The `DocstringGenerator` instance.
            docstring: The existing docstring.

        Returns:
            The summary of the docstring, or `None` if no summary is found.
        """
        if not docstring:
            return None
        stripped = docstring.strip().splitlines()
        return stripped[0].strip() if stripped else None

    def _extract_parameters(self, node: ast.AST) -> List[ParameterDoc]:
        """
        Extracts the parameters from an AST node.

        Args:
            self: The `DocstringGenerator` instance.
            node: The AST node to extract parameters from.

        Returns:
            A list of `ParameterDoc` objects representing the parameters of the node.
        """
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        params: List[ParameterDoc] = []

        def annotation_to_str(annotation: Optional[ast.AST]) -> str:
            """
            Converts an AST annotation to a string.

            Args:
                annotation: The AST annotation to convert.

            Returns:
                A string representation of the annotation, or "Any" if the annotation is None.
            """
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
            isinstance(child, ast.Return) and child.value is not None for child in ast.walk(node)
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
            lines.append(f"{body_indent}{return_info.annotation}: {return_info.description}")

        lines.append(f'{body_indent}"""')
        return [f"{line}\n" for line in lines]

    def _indent_docstring(self, lines: Sequence[str], body_indent: str) -> List[str]:
        """Wrap provided lines inside triple quotes with correct indentation."""

        indented = [f'{body_indent}"""\n']
        for line in lines:
            indented.append(f"{body_indent}{line.rstrip()}\n")
        indented.append(f'{body_indent}"""\n')
        return indented

    def _strip_delimiters(self, text: str) -> str:
        """Remove common Markdown/code fences, language hints, and string delimiters."""

        text = text.strip()
        text = re.sub(r"^```(?:\w+)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        lines = text.splitlines()
        if lines and re.fullmatch(r"\s*(python|py|code)\s*", lines[0], re.IGNORECASE):
            lines = lines[1:]
        text = "\n".join(lines).strip()

        if text.startswith('"""') and text.endswith('"""'):
            return text[3:-3].strip()
        if text.startswith("'''") and text.endswith("'''"):
            return text[3:-3].strip()
        return text

    def _extract_docstring_from_code(self, text: str) -> Optional[str]:
        """Parse code/text and return the first docstring found if present."""

        try:
            tree = ast.parse(text)
        except SyntaxError:
            return None

        root_doc = ast.get_docstring(tree)
        if root_doc:
            return root_doc

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nested = ast.get_docstring(node)
                if nested:
                    return nested
        return None

    def _normalize_llm_output(self, llm_output: str) -> str:
        """Extract a clean docstring body from an LLM response."""

        stripped = self._strip_delimiters(llm_output)
        parsed_docstring = self._extract_docstring_from_code(stripped)
        if parsed_docstring:
            return textwrap.dedent(parsed_docstring).strip()

        literal_match = re.search(r'"""(.*?)"""', stripped, re.DOTALL)
        if literal_match:
            return textwrap.dedent(literal_match.group(1)).strip()

        literal_match = re.search(r"'''(.*?)'''", stripped, re.DOTALL)
        if literal_match:
            return textwrap.dedent(literal_match.group(1)).strip()

        return textwrap.dedent(stripped).strip()

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
        for start, end, replacement, is_replacement in sorted(
            edits, key=lambda item: item[0], reverse=True
        ):
            if is_replacement:
                new_lines[start : end + 1] = replacement
            else:
                new_lines[start:start] = replacement
        return new_lines
