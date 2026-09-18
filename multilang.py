"""Bounded multilingual sink/taint heuristics via tree-sitter.

Full symbolic/import-aware analysis remains Python-only in ``semantic.py``.
This module inventories dangerous sinks and emits advisory ``taint`` findings
when a known source and sink co-occur in the same function, or when a
high-risk sink appears (eval/system/exec-style APIs).
"""

from __future__ import annotations

import re
from pathlib import Path

from source_tools import LANGUAGES

# tree-sitter node types that represent calls / invocations
_CALL_TYPES: dict[str, set[str]] = {
    "c": {"call_expression"},
    "cpp": {"call_expression"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "tsx": {"call_expression"},
    "php": {
        "function_call_expression",
        "method_call_expression",
        "scoped_call_expression",
    },
    "java": {"method_invocation"},
    "go": {"call_expression"},
    "ruby": {"call"},
    "rust": {"call_expression"},
}

# Function-like scopes used to co-locate sources and sinks.
_FUNCTION_TYPES: dict[str, set[str]] = {
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "javascript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    },
    "typescript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    },
    "tsx": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    },
    "php": {"function_definition", "method_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "ruby": {"method", "singleton_method"},
    "rust": {"function_item"},
}

# Callee name → risk. Names are matched case-sensitively on the final identifier.
_SINKS: dict[str, dict[str, str]] = {
    "c": {
        "strcpy": "high",
        "strncpy": "medium",
        "strcat": "high",
        "strncat": "medium",
        "sprintf": "high",
        "vsprintf": "high",
        "gets": "high",
        "scanf": "medium",
        "sscanf": "medium",
        "system": "high",
        "popen": "high",
        "memcpy": "medium",
        "memmove": "medium",
        "execl": "high",
        "execv": "high",
        "execve": "high",
    },
    "cpp": {
        "strcpy": "high",
        "strcat": "high",
        "sprintf": "high",
        "gets": "high",
        "system": "high",
        "popen": "high",
        "memcpy": "medium",
    },
    "javascript": {
        "eval": "high",
        "Function": "high",
        "exec": "high",
        "execSync": "high",
        "spawn": "medium",
        "spawnSync": "medium",
    },
    "typescript": {
        "eval": "high",
        "Function": "high",
        "exec": "high",
        "execSync": "high",
        "spawn": "medium",
        "spawnSync": "medium",
    },
    "tsx": {
        "eval": "high",
        "Function": "high",
        "exec": "high",
        "execSync": "high",
    },
    "php": {
        "eval": "high",
        "assert": "high",
        "system": "high",
        "exec": "high",
        "passthru": "high",
        "shell_exec": "high",
        "popen": "high",
        "proc_open": "high",
        "unserialize": "high",
        "mysqli_query": "medium",
        "mysql_query": "medium",
        "pg_query": "medium",
        "sqlite_query": "medium",
    },
    "java": {
        "exec": "high",
        "execute": "medium",
        "executeQuery": "medium",
        "executeUpdate": "medium",
        "prepareStatement": "medium",
        "createQuery": "medium",
    },
    "go": {
        "Command": "high",
        "Query": "medium",
        "QueryRow": "medium",
        "Exec": "medium",
        "Eval": "high",
    },
    "ruby": {
        "eval": "high",
        "system": "high",
        "exec": "high",
        "spawn": "high",
        "send": "medium",
        "public_send": "medium",
        "constantize": "medium",
    },
    "rust": {
        "Command": "high",
        "from_str_radix": "medium",
    },
}

# Source patterns searched inside a function's source slice (not full dataflow).
_SOURCES: dict[str, tuple[re.Pattern[str], ...]] = {
    "c": (
        re.compile(r"\bargv\b"),
        re.compile(r"\bgetenv\s*\("),
        re.compile(r"\bfgets\s*\("),
        re.compile(r"\bread\s*\("),
        re.compile(r"\bscanf\s*\("),
        re.compile(r"\brecv\s*\("),
    ),
    "cpp": (
        re.compile(r"\bargv\b"),
        re.compile(r"\bgetenv\s*\("),
        re.compile(r"\bstd::cin\b"),
        re.compile(r"\bfgets\s*\("),
    ),
    "javascript": (
        re.compile(r"\breq\.(?:body|query|params|cookies|headers)\b"),
        re.compile(r"\brequest\.(?:body|query|params)\b"),
        re.compile(r"\blocation\.(?:hash|search|href)\b"),
        re.compile(r"\bdocument\.cookie\b"),
        re.compile(r"\bprocess\.argv\b"),
    ),
    "typescript": (
        re.compile(r"\breq\.(?:body|query|params|cookies|headers)\b"),
        re.compile(r"\brequest\.(?:body|query|params)\b"),
        re.compile(r"\bprocess\.argv\b"),
    ),
    "tsx": (
        re.compile(r"\breq\.(?:body|query|params|cookies|headers)\b"),
        re.compile(r"\bprocess\.argv\b"),
    ),
    "php": (
        re.compile(r"\$_(?:GET|POST|REQUEST|COOKIE|SERVER|FILES)\b"),
        re.compile(r"\bfile_get_contents\s*\(\s*['\"]php://input"),
    ),
    "java": (
        re.compile(r"\bgetParameter\s*\("),
        re.compile(r"\bgetHeader\s*\("),
        re.compile(r"\bgetQueryString\s*\("),
        re.compile(r"\bgetInputStream\s*\("),
        re.compile(r"\bargs\b"),
    ),
    "go": (
        re.compile(r"\bos\.Args\b"),
        re.compile(r"\br\.URL\.Query\b"),
        re.compile(r"\br\.FormValue\s*\("),
        re.compile(r"\br\.Header\.Get\s*\("),
        re.compile(r"\bio\.ReadAll\s*\("),
    ),
    "ruby": (
        re.compile(r"\bparams\b"),
        re.compile(r"\brequest\.(?:body|params|headers)\b"),
        re.compile(r"\bENV\b"),
        re.compile(r"\bARGV\b"),
    ),
    "rust": (
        re.compile(r"\benv::args\b"),
        re.compile(r"\benv::var\b"),
        re.compile(r"\bstd::io::stdin\b"),
    ),
}

_SUPPORTED = set(_CALL_TYPES)


def language_for(path: str) -> str | None:
    return LANGUAGES.get(Path(path).suffix.lower())


def _callee_name(node) -> str | None:
    """Best-effort final identifier of a call/invocation node."""
    for field in ("name", "function", "method"):
        child = node.child_by_field_name(field)
        if child is not None:
            text = child.text.decode("utf-8", errors="replace").strip()
            parts = re.split(r"::|->|\.|\\", text)
            name = re.sub(r"[^\w$]", "", parts[-1].strip()) if parts else ""
            if name:
                return name
    text = node.text.decode("utf-8", errors="replace")
    # Last identifier immediately before an argument list (handles a.b().c(x)).
    matches = list(re.finditer(r"([A-Za-z_$][\w$]*)\s*\(", text))
    if matches:
        return matches[-1].group(1)
    head = text.split("(", 1)[0].strip()
    if not head:
        return None
    parts = re.split(r"::|->|\.|\\", head)
    name = re.sub(r"[^\w$]", "", parts[-1].strip()) if parts else ""
    return name or None


def _line_of(node) -> int:
    return int(node.start_point[0]) + 1


def _collect_nodes(root, types: set[str]) -> list:
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in types:
            out.append(node)
        stack.extend(node.children)
    return out


def analyze_file(path: str, source: str, *, disabled=()) -> tuple[list[dict], list[str]]:
    """Analyze one non-Python source file. Returns findings and warnings."""
    disabled = set(disabled)
    language = language_for(path)
    if language is None:
        return [], [f"{path}: unsupported language for multilingual analysis"]
    if language not in _SUPPORTED:
        return [], [f"{path}: no multilingual sink rules for {language}"]
    if language == "python":
        return [], []

    from tree_sitter_language_pack import get_parser

    data = source.encode("utf-8")
    tree = get_parser(language).parse(data)
    warnings = []
    if tree.root_node.has_error:
        warnings.append(f"{path}: syntax errors; multilingual analysis incomplete")

    sinks = _SINKS.get(language, {})
    source_pats = _SOURCES.get(language, ())
    call_types = _CALL_TYPES[language]
    function_types = _FUNCTION_TYPES.get(language, set())

    findings: list[dict] = []
    calls = _collect_nodes(tree.root_node, call_types)
    functions = _collect_nodes(tree.root_node, function_types) if function_types else []

    # Map each call to an enclosing function byte range when possible.
    def enclosing(node):
        for fn in functions:
            if fn.start_byte <= node.start_byte and node.end_byte <= fn.end_byte:
                return fn
        return None

    # Per-function source presence cache
    fn_has_source: dict[int, bool] = {}
    for fn in functions:
        slice_text = data[fn.start_byte : fn.end_byte].decode("utf-8", errors="replace")
        fn_has_source[id(fn)] = any(p.search(slice_text) for p in source_pats)

    # File-level source fallback when there is no function scope (scripts).
    file_has_source = any(p.search(source) for p in source_pats)

    for call in calls:
        name = _callee_name(call)
        if not name or name not in sinks:
            continue
        risk = sinks[name]
        line = _line_of(call)
        findings.append(
            dict(
                kind="sink",
                path=path,
                line=line,
                detail=f"Call to {name}",
                witness=None,
            )
        )
        if "taint" in disabled:
            continue
        fn = enclosing(call)
        colocated = fn_has_source.get(id(fn), False) if fn is not None else file_has_source
        if colocated:
            findings.append(
                dict(
                    kind="taint",
                    path=path,
                    line=line,
                    detail=(
                        f"Heuristic: known source and sink `{name}` co-occur in the "
                        f"same {language} function (not a symbolic proof)"
                    ),
                    witness=None,
                )
            )
        elif risk == "high":
            findings.append(
                dict(
                    kind="taint",
                    path=path,
                    line=line,
                    detail=(
                        f"Heuristic: high-risk {language} sink `{name}` "
                        f"(not a symbolic proof)"
                    ),
                    witness=None,
                )
            )

    unique = {repr(f): f for f in findings}
    return list(unique.values()), warnings


def analyze_files(files: dict[str, str], *, disabled=()) -> tuple[list[dict], list[str]]:
    """Analyze all non-Python files in ``files``."""
    findings: list[dict] = []
    warnings: list[str] = []
    analyzed = 0
    for path, source in files.items():
        language = language_for(path)
        if language in {None, "python"}:
            continue
        file_findings, file_warnings = analyze_file(path, source, disabled=disabled)
        findings.extend(file_findings)
        warnings.extend(file_warnings)
        if language in _SUPPORTED:
            analyzed += 1
    if analyzed:
        warnings.append(
            f"{analyzed} non-Python files: multilingual sink/taint heuristics applied; "
            "full symbolic/import analysis remains Python-only"
        )
    return findings, warnings
