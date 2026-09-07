"""Compiler / heuristic program-analysis lane (injection-immune).

Lane B combines:
  1. Python AST analysis (sinks, taint, light symbolic, cross-file)
  2. Diff-aware, multi-language heuristics over the unified PR diff
     (removed sanitizers / weakened bounds / introduced sinks)

NL narrative is never trusted — only code and diff hunks.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Iterable


def normalize_source(source: str) -> str:
    """Dedent a source snippet for AST parsing."""
    return textwrap.dedent(source.replace("\t", "    ")).strip() + "\n"


def strip_comments_and_docstrings(source: str) -> str:
    """Return source with docstrings and comments removed."""
    source = normalize_source(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(re.sub(r"#.*$", "", line) for line in source.splitlines())

    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module),
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] if len(body) > 1 else [ast.Pass()]

    return ast.unparse(tree)


DANGEROUS_SINKS = {
    "eval",
    "exec",
    "compile",
    "system",
    "popen",
    "execute",
    "loads",
    "__import__",
}

TAINT_SOURCES = {
    "input",
    "getenv",
    "get",
    "post",
    "form",
    "args",
    "readline",
    "read",
    "recv",
    "chat",
    "generate",
}


@dataclass
class Finding:
    kind: str
    path: str
    line: int | None
    detail: str


@dataclass
class AnalysisReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def has_blocking_issue(self) -> bool:
        # Hard blocks come from diff-aware multi-lang signals.
        blocking = {
            "diff_sink",
            "diff_csrf_removed",
            "diff_authz_removed",
        }
        return any(finding.kind in blocking for finding in self.findings)

    @property
    def has_advisory_issue(self) -> bool:
        advisory = {
            "diff_sanitizer_removed",
            "diff_bounds_weakened",
            "sink",
            "taint",
            "symbolic",
            "cross_file",
        }
        return any(finding.kind in advisory for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocking": self.has_blocking_issue,
            "findings": [
                {
                    "kind": finding.kind,
                    "path": finding.path,
                    "line": finding.line,
                    "detail": finding.detail,
                }
                for finding in self.findings
            ],
        }


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def find_risky_sinks(source: str, path: str = "<string>") -> list[Finding]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in DANGEROUS_SINKS:
                findings.append(
                    Finding(
                        kind="sink",
                        path=path,
                        line=getattr(node, "lineno", None),
                        detail=f"dangerous call `{name}()`",
                    )
                )
    return findings


def find_taint_flows(source: str, path: str = "<string>") -> list[Finding]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    tainted: set[str] = set()
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = _call_name(node.value)
            if name in TAINT_SOURCES:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tainted.add(target.id)

        if isinstance(node, ast.Call):
            sink = _call_name(node)
            if sink in DANGEROUS_SINKS:
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in tainted:
                        findings.append(
                            Finding(
                                kind="taint",
                                path=path,
                                line=getattr(node, "lineno", None),
                                detail=f"tainted `{arg.id}` reaches `{sink}()`",
                            )
                        )
    return findings


def find_symbolic_reachability(source: str, path: str = "<string>") -> list[Finding]:
    """Lightweight symbolic note: sinks under always-true / unconstrained guards."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    findings: list[Finding] = []

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            always_true = isinstance(node.test, ast.Constant) and bool(node.test.value)
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = _call_name(child)
                    if name in DANGEROUS_SINKS and always_true:
                        findings.append(
                            Finding(
                                kind="symbolic",
                                path=path,
                                line=getattr(child, "lineno", None),
                                detail=(
                                    f"`{name}()` reachable under always-true guard"
                                ),
                            )
                        )
            self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def resolve_cross_file_symbols(
    files: dict[str, str],
) -> list[Finding]:
    """Flag sinks whose callee is defined in another changed file."""
    definitions: dict[str, str] = {}
    for path, source in files.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[node.name] = path

    findings: list[Finding] = []
    for path, source in files.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name in definitions and definitions[name] != path:
                remote = files[definitions[name]]
                sink_pattern = r"\b(" + "|".join(sorted(DANGEROUS_SINKS)) + r")\b"
                if re.search(sink_pattern, remote):
                    findings.append(
                        Finding(
                            kind="cross_file",
                            path=path,
                            line=getattr(node, "lineno", None),
                            detail=(
                                f"call `{name}()` resolves to {definitions[name]} "
                                "which contains dangerous sinks"
                            ),
                        )
                    )
    return findings


# ---------------------------------------------------------------------------
# Diff-aware multi-language heuristics
# ---------------------------------------------------------------------------

@dataclass
class DiffHunkFile:
    path: str
    added: list[tuple[int | None, str]]  # (new_lineno, text)
    removed: list[tuple[int | None, str]]  # (old_lineno, text)


def parse_unified_diff(diff: str) -> list[DiffHunkFile]:
    """Parse a unified diff into per-file added/removed lines."""
    files: list[DiffHunkFile] = []
    current: DiffHunkFile | None = None
    old_line: int | None = None
    new_line: int | None = None

    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            current = None
            continue
        if raw.startswith("+++ "):
            path = raw[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path == "/dev/null":
                current = None
                continue
            current = DiffHunkFile(path=path, added=[], removed=[])
            files.append(current)
            continue
        if current is None:
            continue
        if raw.startswith("@@"):
            match = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
            if match:
                old_line = int(match.group(1))
                new_line = int(match.group(2))
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            current.added.append((new_line, raw[1:]))
            if new_line is not None:
                new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            current.removed.append((old_line, raw[1:]))
            if old_line is not None:
                old_line += 1
        else:
            # context line
            if old_line is not None:
                old_line += 1
            if new_line is not None:
                new_line += 1
    return files


def _normalize_cwe(cwe_id: str | None) -> str | None:
    if not cwe_id:
        return None
    match = re.search(r"(\d+)", str(cwe_id))
    return match.group(1) if match else None


# (kind, cwe_ids or None=all, pattern, detail_template)
_REMOVED_SANITIZER_RULES: list[tuple[str, set[str] | None, re.Pattern[str], str]] = [
    (
        "diff_sanitizer_removed",
        {"89"},
        re.compile(
            r"quote(?:Identifier)?\s*\(|mysqli_real_escape|pg_escape|"
            r"addslashes\s*\(|PDO::quote|sqlalchemy\.text|"
            r"parameteriz|bind_param|prepared\s*statement",
            re.I,
        ),
        "removed SQL quoting / parameterization",
    ),
    (
        "diff_sanitizer_removed",
        {"79"},
        re.compile(
            r"escape(?:html|Html|HTML)?|htmlspecialchars|htmlentities|"
            r"sanitize|DOMPurify|xss|encodeURIComponent|"
            r"bleach\.clean|markupsafe\.escape|cgi\.escape",
            re.I,
        ),
        "removed HTML/XSS escaping or sanitizer",
    ),
    (
        "diff_sanitizer_removed",
        {"78"},
        re.compile(
            r"pipes\.quote|shlex\.quote|escapeshell(?:arg|cmd)|"
            r"Shellwords\.escape|subprocess\.list2cmdline",
            re.I,
        ),
        "removed shell argument escaping",
    ),
    (
        "diff_sanitizer_removed",
        {"416"},
        re.compile(r"ManuallyDrop|drop_in_place|Pin::new|mem::forget", re.I),
        "removed memory-safety / lifetime guard",
    ),
    (
        "diff_sanitizer_removed",
        {"22"},
        re.compile(
            r"realpath|canonical|Path\.resolve|filepath\.Clean|"
            r"os\.path\.normpath|basename\s*\(|secure_filename|"
            r"path\.normalize|HasPrefix\s*\(.*\.+\.|strings\.Contains\s*\(.*\.\.",
            re.I,
        ),
        "removed path canonicalization / traversal guard",
    ),
    (
        "diff_csrf_removed",
        {"352"},
        re.compile(
            r"protect_from_forgery|csrf_protect|verify_csrf|"
            r"csrf_token|CsrfViewMiddleware|@csrf",
            re.I,
        ),
        "removed CSRF protection",
    ),
    (
        "diff_authz_removed",
        {"862"},
        re.compile(
            r"authorize|authoris|permission_required|can\?|"
            r"require_.*auth|check_auth|is_admin|ensure_.*user|"
            r"@login_required|authenticate!",
            re.I,
        ),
        "removed authorization / auth check",
    ),
]

_ADDED_SINK_RULES: list[tuple[str, set[str] | None, re.Pattern[str], str]] = [
    (
        "diff_sink",
        {"89"},
        re.compile(
            r"(SELECT|INSERT|UPDATE|DELETE|WHERE).{0,80}(\+|'\s*\.|\"\s*\+|"
            r"\$\w+|%\s*\(|\.format\s*\(|f[\"'])",
            re.I,
        ),
        "SQL keyword concatenated / interpolated with untrusted-looking data",
    ),
    (
        "diff_sink",
        {"89"},
        re.compile(
            r"`[^`]*\$\w+[^`]*`\s*(?:BETWEEN|LIKE|=|<|>)|"
            r"'\s*\.\s*\$\w+\s*\.\s*'|"
            r"\"\s*\+\s*\w+\s*\+\s*\"",
            re.I,
        ),
        "identifier/value spliced into SQL via string concat",
    ),
    (
        "diff_sink",
        {"78"},
        re.compile(
            r"\b(?:os\.system|subprocess\.(?:call|run|Popen)|popen|system|exec|"
            r"Runtime\.exec|ProcessBuilder|child_process|execSync|spawn\s*\(|"
            r"Kernel\.system|IO\.popen|open3)\b|"
            r"`[^`]*\$\{?\w+",
            re.I,
        ),
        "command-execution sink introduced",
    ),
    (
        "diff_sink",
        {"79"},
        re.compile(
            r"innerHTML|outerHTML|dangerouslySetInnerHTML|"
            r"document\.write|v-html|\[innerHTML\]|"
            r"echo\s+\$_(?:GET|POST|REQUEST)|"
            r"\{\{\s*\w+\s*\}\}(?!\s*\||\s*escape)",
            re.I,
        ),
        "XSS sink / unescaped HTML output introduced",
    ),
    (
        "diff_sink",
        {"22"},
        re.compile(
            r"(?:open|readFile|writeFile|include|require|fopen|file_get_contents|"
            r"os\.Remove|os\.Open|ioutil\.ReadFile)\s*\([^)]*(?:\+|\$\{|/?\.\./)",
            re.I,
        ),
        "filesystem API with concatenated / traversal-prone path",
    ),
    (
        "diff_sink",
        {"22"},
        re.compile(
            r"\b(?:Mkdir|mkdir|os\.mkdir|makedirs)\s*\([^)]*0[0-7]*[2367][0-7]*",
            re.I,
        ),
        "directory created with world/group-writable mode",
    ),
    (
        "diff_sink",
        {"94"},
        re.compile(
            r"\b(?:eval|exec|Function\s*\(|setTimeout\s*\(\s*['\"]|"
            r"setInterval\s*\(\s*['\"]|compile\s*\(|__import__|"
            r"yaml\.load\s*\(|pickle\.loads|Marshal\.load|"
            r"unserialize\s*\()\b",
            re.I,
        ),
        "code-execution / unsafe deserialization sink introduced",
    ),
    (
        "diff_sink",
        {"416"},
        re.compile(
            r"drop_in_place|ManuallyDrop|use.after.free|dangling\s+pointer|"
            r"\bfree\s*\(",
            re.I,
        ),
        "use-after-free / manual-drop risk pattern",
    ),
    (
        "diff_sink",
        {"787", "125"},
        re.compile(
            r"\b(?:memcpy|strcpy|strcat|gets|sprintf)\s*\(",
            re.I,
        ),
        "unchecked buffer write / classic overflow API",
    ),
]

_BOUNDS_WEAKEN = re.compile(
    r"if\s*\([^)]*(?:\|\||&&)?[^)]*(?:>=?|<=?)\s*(?:NR_\w+|SIZE|\w+_MAX|sizeof)",
    re.I,
)


def _rule_applies(cwes: set[str] | None, target: str | None) -> bool:
    if cwes is None or target is None:
        return True
    return target in cwes


def _clip(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def analyze_diff(
    diff: str,
    *,
    cwe_id: str | None = None,
) -> list[Finding]:
    """Language-agnostic findings from a unified diff."""
    if not (diff or "").strip():
        return []

    target = _normalize_cwe(cwe_id)
    findings: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()

    def add(kind: str, path: str, line: int | None, detail: str) -> None:
        key = (kind, path, detail)
        if key in seen:
            return
        seen.add(key)
        findings.append(Finding(kind=kind, path=path, line=line, detail=detail))

    for file in parse_unified_diff(diff):
        removed_text = "\n".join(text for _, text in file.removed)
        added_text = "\n".join(text for _, text in file.added)

        for kind, cwes, pattern, detail in _REMOVED_SANITIZER_RULES:
            if not _rule_applies(cwes, target):
                continue
            removed_hits = [(ln, t) for ln, t in file.removed if pattern.search(t)]
            added_hits = [t for _, t in file.added if pattern.search(t)]
            # Net removal only — benign fixes often replace one sanitizer with another.
            if removed_hits and len(removed_hits) > len(added_hits):
                ln, text = removed_hits[0]
                add(
                    kind,
                    file.path,
                    ln,
                    f"{detail}: `{_clip(text)}`",
                )

        # Bounds: require removing a compound/upper-bound guard and adding a weaker if.
        if target in {None, "125", "787"}:
            for ln, text in file.removed:
                if not re.search(r"\bif\s*\(", text):
                    continue
                has_upper = bool(
                    re.search(
                        r"(?:NR_\w+|SIZE|sizeof|_MAX|_SZ|length|count).*"
                        r"(?:>=|>|<=|<)|(?:>=|>|<=|<).*(?:NR_\w+|SIZE|sizeof|_MAX|_SZ)",
                        text,
                        re.I,
                    )
                ) or ("NR_syscalls" in text)
                compound = bool(re.search(r"\|\||&&", text))
                if not (has_upper or compound):
                    continue
                # Added ifs should not still contain the same upper-bound token.
                tokens = re.findall(r"NR_\w+|\w+_MAX|\w+_SZ|sizeof\s*\([^)]*\)", text)
                added_ifs = [t for _, t in file.added if re.search(r"\bif\s*\(", t)]
                if not added_ifs:
                    continue
                if tokens and any(
                    any(tok in a for tok in tokens) for a in added_ifs
                ):
                    continue
                # Weaker: added condition text shorter / fewer conjuncts.
                if any(len(a) < len(text) or a.count("&&") + a.count("||") < (
                    text.count("&&") + text.count("||")
                ) for a in added_ifs):
                    add(
                        "diff_bounds_weakened",
                        file.path,
                        ln,
                        f"bounds check weakened: `{_clip(text)}`",
                    )
                    break

        for kind, cwes, pattern, detail in _ADDED_SINK_RULES:
            if not _rule_applies(cwes, target):
                continue
            for line_no, text in file.added:
                if pattern.search(text):
                    if any(text == rtext for _, rtext in file.removed):
                        continue
                    add(
                        kind,
                        file.path,
                        line_no,
                        f"{detail}: `{_clip(text)}`",
                    )
                    break

        if target in {None, "352"}:
            for ln, text in file.removed:
                if re.search(r"protect_from_forgery|csrf_protect|verify_csrf", text, re.I):
                    if not re.search(
                        r"protect_from_forgery|csrf_protect|verify_csrf",
                        added_text,
                        re.I,
                    ):
                        add(
                            "diff_csrf_removed",
                            file.path,
                            ln,
                            f"CSRF protection deleted: `{_clip(text)}`",
                        )

    return findings


class ProgramAnalyzer:
    """Lane B: deterministic analysis over PR file contents and unified diff."""

    def analyze(
        self,
        files: dict[str, str] | None = None,
        *,
        diff: str | None = None,
        cwe_id: str | None = None,
    ) -> AnalysisReport:
        report = AnalysisReport()
        files = files or {}

        if files:
            py_files = {
                path: normalize_source(source)
                for path, source in files.items()
                if path.endswith((".py", ".pyi"))
            }
            for path, source in py_files.items():
                report.findings.extend(find_risky_sinks(source, path))
                report.findings.extend(find_taint_flows(source, path))
                report.findings.extend(find_symbolic_reachability(source, path))
            report.findings.extend(resolve_cross_file_symbols(py_files))

        if diff:
            report.findings.extend(analyze_diff(diff, cwe_id=cwe_id))

        return report
