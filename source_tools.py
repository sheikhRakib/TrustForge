"""Syntax-aware source preparation. Preserve original file and line coordinates."""

from __future__ import annotations
import ast
from pathlib import Path

LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".php": "php",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".cs": "csharp",
    ".sh": "bash",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
}


def strip_source(
    source: str, path: str, *, mask_literal_text: bool = False
) -> tuple[str, list[str]]:
    """Blank syntax comment nodes and Python docstrings; never regex-edit strings.

    Unsupported or malformed syntax is returned with an explicit coverage warning.
    Known comments in an error-recovering syntax tree are still removed.
    Diff heuristics may mask literal text too; executable interpolation is kept.
    Auditors use the default, which preserves string values.
    """
    language = LANGUAGES.get(Path(path).suffix.lower())
    if not language:
        return source, [f"{path}: unsupported syntax for stripping"]
    from tree_sitter_language_pack import get_parser

    data = source.encode("utf-8")
    tree = get_parser(language).parse(data)
    ranges = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        literal_text = node.type in {
            "string_content",
            "string_fragment",
            "raw_string_literal_content",
            "interpreted_string_literal_content",
            "heredoc_content",
            "escape_sequence",
            "string_start",
            "string_end",
        }
        terminal_literal = not node.named_children and (
            "string" in node.type or node.type in {"char_literal", "character_literal"}
        )
        if "comment" in node.type or (
            mask_literal_text and (literal_text or terminal_literal)
        ):
            ranges.append((node.start_byte, node.end_byte))
        else:
            stack.extend(node.children)
    warnings = (
        [f"{path}: syntax errors; stripping coverage incomplete"]
        if tree.root_node.has_error
        else []
    )
    if language == "python":
        try:
            parsed = ast.parse(source)
        except SyntaxError:
            pass
        else:
            starts = [0]
            for line in data.splitlines(keepends=True):
                starts.append(starts[-1] + len(line))
            for node in ast.walk(parsed):
                if isinstance(
                    node,
                    (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    if node.body and isinstance(node.body[0], ast.Expr):
                        expr = node.body[0]
                        if isinstance(expr.value, ast.Constant) and isinstance(
                            expr.value.value, str
                        ):
                            ranges.append(
                                (
                                    starts[expr.lineno - 1] + expr.col_offset,
                                    starts[expr.end_lineno - 1] + expr.end_col_offset,
                                )
                            )
    out = bytearray(data)
    for start, end in ranges:
        for i in range(start, end):
            if out[i] not in (10, 13):
                out[i] = 32
    return out.decode("utf-8"), warnings


def strip_comments_and_docstrings(source: str, path: str = "snippet.py") -> str:
    return strip_source(source, path)[0]
