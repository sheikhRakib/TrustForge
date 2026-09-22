"""Deterministic, bounded Python semantics and multilingual analysis."""

from __future__ import annotations
import re
import threading
from dataclasses import dataclass, field
from source_tools import strip_source, LANGUAGES
from semantic import PythonAnalyzer
from multilang import analyze_files as analyze_multilang_files

_SEMANTIC_LOCK = threading.Lock()


@dataclass
class Finding:
    kind: str
    path: str
    line: int | None
    detail: str
    witness: dict | None = None


@dataclass
class AnalysisReport:
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_advisory_issue(self):
        return any(
            f.kind
            in {
                "taint",
                "symbolic",
                "diff_sink",
                "diff_csrf_removed",
                "diff_authz_removed",
                "diff_sanitizer_removed",
                "diff_bounds_weakened",
            }
            for f in self.findings
        )

    def to_dict(self):
        from dataclasses import asdict

        return {
            "advisory": self.has_advisory_issue,
            "findings": [asdict(f) for f in self.findings],
            "warnings": self.warnings,
        }


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
            r"(?:\.\s*(?:execute|executemany|query)|mysqli_query|pg_query)"
            r"\s*\([^;\n]*(?:\+|\.format\s*\(|\s\.\s)",
            re.I,
        ),
        "SQL API argument constructed with concatenation / formatting",
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
        from pathlib import Path

        if Path(file.path).suffix.lower() not in LANGUAGES:
            continue
        # Reconstruct hunk-side text including context before syntax stripping.
        # This preserves multiline comments within each available patch hunk.
        sides = {"old": [], "new": []}
        active = False
        for raw in diff.splitlines():
            if raw.startswith("+++ "):
                active = raw[4:].removeprefix("b/").strip() == file.path
            elif raw.startswith("diff --git "):
                active = False
            elif active and not raw.startswith(("@@", "\\")):
                if raw.startswith("-"):
                    sides["old"].append(raw[1:])
                elif raw.startswith("+"):
                    sides["new"].append(raw[1:])
                elif raw.startswith(" "):
                    sides["old"].append(raw[1:])
                    sides["new"].append(raw[1:])
        maps = {}
        for side in ("old", "new"):
            raw_lines = sides[side]
            cleaned, _ = strip_source(
                "\n".join(raw_lines), file.path, mask_literal_text=True
            )
            maps[side] = {}
            for raw, clean in zip(raw_lines, cleaned.split("\n")):
                # Keep duplicates conservative: only inspect text surviving stripping.
                maps[side].setdefault(raw, []).append(clean)
        file.removed = [
            (ln, next(iter(maps["old"].get(t, [""])), "")) for ln, t in file.removed
        ]
        file.added = [
            (ln, next(iter(maps["new"].get(t, [""])), "")) for ln, t in file.added
        ]
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
                if tokens and any(any(tok in a for tok in tokens) for a in added_ifs):
                    continue
                # Weaker: added condition text shorter / fewer conjuncts.
                if any(
                    len(a) < len(text)
                    or a.count("&&") + a.count("||")
                    < (text.count("&&") + text.count("||"))
                    for a in added_ifs
                ):
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
                if re.search(
                    r"protect_from_forgery|csrf_protect|verify_csrf", text, re.I
                ):
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
    """Analyze supplied source context; unsupported semantics are explicit."""

    def analyze(self, files=None, *, diff=None, cwe_id=None, disabled=()):
        files = files or {}
        report = AnalysisReport()
        # Z3 default contexts are not thread-safe for concurrent analysis.
        with _SEMANTIC_LOCK:
            findings, warnings = PythonAnalyzer(files, disabled=disabled).run()
        multi_findings, multi_warnings = analyze_multilang_files(
            files, disabled=disabled
        )
        findings = list(findings) + multi_findings
        warnings = list(warnings) + multi_warnings
        report.findings = [Finding(**f) for f in findings if f["kind"] not in disabled]
        report.warnings = warnings
        if diff and "diff" not in disabled:
            # Never condition the defense on the benchmark's ground-truth CWE.
            report.findings.extend(analyze_diff(diff))
        return report
