"""Unified source semantics: Python analysis and bounded multilingual rules.

No submitted code is executed. Z3 produces witnesses for supported paths.
Unsupported operations, loops, dynamic dispatch, and exhausted bounds are recorded.
"""

from __future__ import annotations
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import z3
from tree_sitter_language_pack import get_parser
from source_tools import LANGUAGES, MAX_ANALYSIS_SOURCE_BYTES

SOURCES = {
    "input",
    "os.getenv",
    "os.environ.get",
    "request.args.get",
    "request.form.get",
    "request.GET.get",
    "request.POST.get",
    "flask.request.args.get",
    "flask.request.form.get",
    "sys.stdin.readline",
    "socket.recv",
}
SINKS = {
    "eval",
    "exec",
    "os.system",
    "os.popen",
    "pickle.loads",
    "yaml.load",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "cursor.execute",
    "db.execute",
}


@dataclass
class Value:
    expr: object = None
    tainted: bool = False
    symbol: str | None = None


class PythonAnalyzer:
    def __init__(self, files, max_steps=3000, max_depth=4, disabled=()):
        self.disabled = set(disabled)
        self.files = files
        self.modules = {}
        self.functions = {}
        self.imports = {}
        self.findings = []
        self.warnings = set()
        self.steps = 0
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.inputs = {}
        self.input_counter = 0
        for path, source in files.items():
            if not path.endswith((".py", ".pyi")):
                continue
            module = str(PurePosixPath(path).with_suffix("")).replace("/", ".")
            if module.endswith(".__init__"):
                module = module[:-9]
            try:
                tree = ast.parse(source)
            except SyntaxError:
                self.warnings.add(f"{path}: Python parse failed")
                continue
            self.modules[module] = (path, tree)
            aliases = {}
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        aliases[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0]
                        )
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        package = (
                            module.split(".")
                            if path.endswith("__init__.py")
                            else module.split(".")[:-1]
                        )
                        base = ".".join(
                            package[: len(package) - node.level + 1]
                            + ([base] if base else [])
                        )
                    for alias in node.names:
                        aliases[alias.asname or alias.name] = (
                            f"{base}.{alias.name}".strip(".")
                        )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.functions[f"{module}.{node.name}"] = (module, node)
            self.imports[module] = aliases

    @property
    def taint_enabled(self):
        return "taint" not in self.disabled

    @property
    def symbolic_enabled(self):
        # Symbolic witnesses are defined only for tainted source-to-sink paths.
        return self.taint_enabled and "symbolic" not in self.disabled

    def name(self, node, module, env):
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id].symbol or "<local>"
            return self.imports[module].get(
                node.id,
                f"{module}.{node.id}"
                if f"{module}.{node.id}" in self.functions
                else node.id,
            )
        if isinstance(node, ast.Attribute):
            return self.name(node.value, module, env) + "." + node.attr
        return "<dynamic>"

    def fresh(self, node, module, integer=False):
        if not self.taint_enabled:
            return Value()
        self.input_counter += 1
        key = f"{module}:line{node.lineno}:input{self.input_counter}"
        expr = None
        if self.symbolic_enabled:
            expr = z3.Int(key) if integer else z3.String(key)
            self.inputs[str(expr)] = expr
        return Value(expr, True)

    def eval(self, node, env, module, conditions, depth):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return Value(z3.BoolVal(node.value))
            if isinstance(node.value, int):
                return Value(z3.IntVal(node.value))
            if isinstance(node.value, str):
                return Value(z3.StringVal(node.value))
            return Value()
        if isinstance(node, ast.Name):
            return env.get(node.id, Value(symbol=self.name(node, module, env)))
        if isinstance(node, ast.Call):
            name = self.name(node.func, module, env)
            if name in SOURCES:
                return self.fresh(node, module)
            if (
                name == "int"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and self.name(node.args[0].func, module, env) in SOURCES
            ):
                return self.fresh(node, module, True)
            args = [self.eval(a, env, module, conditions, depth) for a in node.args]
            keyword_values = [
                self.eval(a.value, env, module, conditions, depth)
                for a in node.keywords
            ]
            args += keyword_values
            taint = any(a.tainted for a in args)
            if name in SINKS or name.endswith((".execute", ".executemany")):
                sink_args = args
                if name.endswith((".execute", ".executemany")):
                    # DB-API bound values do not become SQL syntax. Inspect the
                    # query argument, while still evaluating every argument above.
                    sink_args = (
                        args[:1]
                        if node.args
                        else [
                            value
                            for keyword, value in zip(node.keywords, keyword_values)
                            if keyword.arg in {"query", "sql", "statement", "operation"}
                        ]
                    )
                    if not sink_args or any(k.arg is None for k in node.keywords):
                        self.warnings.add(
                            f"{module}:{node.lineno}: SQL query argument binding incomplete"
                        )
                    taint = any(a.tainted for a in sink_args)
                path = self.modules[module][0]
                self.findings.append(
                    dict(
                        kind="sink",
                        path=path,
                        line=node.lineno,
                        detail=f"Call to {name}",
                        witness=None,
                    )
                )
                if taint and self.taint_enabled:
                    self.findings.append(
                        dict(
                            kind="taint",
                            path=path,
                            line=node.lineno,
                            detail=f"Untrusted data reaches {name}",
                            witness=None,
                        )
                    )
                    if self.symbolic_enabled and conditions is not None and all(
                        not a.tainted or a.expr is not None for a in sink_args
                    ):
                        solver = z3.Solver()
                        solver.set(timeout=100)
                        solver.add(*conditions)
                        status = solver.check()
                        if status == z3.sat:
                            model = solver.model()
                            witness = {
                                key: str(model.eval(expr, model_completion=True))
                                for key, expr in self.inputs.items()
                                if any(str(expr) in str(c) for c in conditions)
                                or any(
                                    a.expr is not None and str(expr) in str(a.expr)
                                    for a in sink_args
                                )
                            }
                            self.findings.append(
                                dict(
                                    kind="symbolic",
                                    path=path,
                                    line=node.lineno,
                                    detail=f"Satisfiable modeled path to {name}; witness is not an exploit proof",
                                    witness=witness,
                                )
                            )
                        elif status == z3.unknown:
                            self.warnings.add(f"{path}: solver timeout/unknown")
                return Value(None, taint)
            if name in self.functions:
                if depth >= self.max_depth:
                    self.warnings.add(f"{module}: call-depth limit")
                    return Value(None, taint)
                remote, fn = self.functions[name]
                if remote != module and "cross_file" in self.disabled:
                    self.warnings.add(f"{module}: cross-file resolution disabled")
                    return Value(None, taint)
                if remote != module:
                    self.findings.append(
                        dict(
                            kind="cross_file",
                            path=self.modules[module][0],
                            line=node.lineno,
                            detail=f"{name} resolves to {self.modules[remote][0]}:{fn.lineno}",
                            witness=None,
                        )
                    )
                bound = {
                    p.arg: (args[i] if i < len(args) else Value())
                    for i, p in enumerate(fn.args.args)
                }
                if node.keywords or fn.args.vararg or fn.args.kwarg:
                    self.warnings.add(
                        f"{module}: unsupported keyword/variadic call binding"
                    )
                    return Value(None, taint)
                states = self.walk(fn.body, bound, remote, conditions, depth + 1)
                returns = [e.get("__return__", Value()) for e, _, _ in states]
                if len(returns) == 1:
                    return returns[0]
                return Value(None, any(v.tainted for v in returns))
            if (
                name in {"str", "int", "len"}
                and len(args) == 1
                and args[0].expr is not None
            ):
                a = args[0]
                if name == "len" and z3.is_string(a.expr):
                    return Value(z3.Length(a.expr), a.tainted)
                if name == "int" and z3.is_int(a.expr):
                    return a
                if name == "str" and z3.is_string(a.expr):
                    return a
            self.warnings.add(f"{module}:{node.lineno}: unsupported call {name}")
            return Value(None, taint)
        if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp, ast.UnaryOp)):
            children = (
                [node.left, node.right]
                if isinstance(node, ast.BinOp)
                else [node.left, *node.comparators]
                if isinstance(node, ast.Compare)
                else node.values
                if isinstance(node, ast.BoolOp)
                else [node.operand]
            )
            vals = [self.eval(n, env, module, conditions, depth) for n in children]
            taint = any(v.tainted for v in vals)
            if all(v.expr is not None for v in vals):
                a = [v.expr for v in vals]
                try:
                    if isinstance(node, ast.BinOp):
                        op = node.op
                        if isinstance(op, ast.Add):
                            return Value(a[0] + a[1], taint)
                        if isinstance(op, ast.Sub):
                            return Value(a[0] - a[1], taint)
                        if isinstance(op, ast.Mult):
                            return Value(a[0] * a[1], taint)
                    if isinstance(node, ast.Compare):
                        ops = {
                            ast.Eq: lambda x, y: x == y,
                            ast.NotEq: lambda x, y: x != y,
                            ast.Lt: lambda x, y: x < y,
                            ast.LtE: lambda x, y: x <= y,
                            ast.Gt: lambda x, y: x > y,
                            ast.GtE: lambda x, y: x >= y,
                        }
                        return Value(
                            z3.And(
                                *[
                                    ops[type(o)](a[i], a[i + 1])
                                    for i, o in enumerate(node.ops)
                                ]
                            ),
                            taint,
                        )
                    if isinstance(node, ast.BoolOp):
                        return Value(
                            (z3.And if isinstance(node.op, ast.And) else z3.Or)(*a),
                            taint,
                        )
                    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                        return Value(z3.Not(a[0]), taint)
                    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                        return Value(-a[0], taint)
                except (z3.Z3Exception, KeyError, TypeError):
                    pass
            self.warnings.add(
                f"{module}:{node.lineno}: unsupported symbolic expression"
            )
            return Value(None, taint)
        vals = [
            self.eval(n, env, module, conditions, depth)
            for n in ast.iter_child_nodes(node)
            if isinstance(n, ast.expr)
        ]
        self.warnings.add(
            f"{module}:{getattr(node, 'lineno', 0)}: unsupported expression {type(node).__name__}"
        )
        return Value(None, any(v.tainted for v in vals))

    def walk(self, body, env, module, conditions, depth=0):
        states = [(dict(env), conditions, False)]
        for node in body:
            out = []
            for env, cond, stopped in states:
                if stopped:
                    out.append((env, cond, True))
                    continue
                self.steps += 1
                if self.steps > self.max_steps:
                    self.warnings.add("symbolic step limit reached")
                    return states
                if isinstance(
                    node,
                    (
                        ast.Import,
                        ast.ImportFrom,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                    ),
                ):
                    out.append((env, cond, False))
                    continue
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    if node.value is not None:
                        value = self.eval(node.value, env, module, cond, depth)
                        for target in (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        ):
                            if isinstance(target, ast.Name):
                                env[target.id] = value
                            else:
                                self.warnings.add(
                                    f"{module}: unsupported assignment target"
                                )
                elif isinstance(node, ast.Expr):
                    self.eval(node.value, env, module, cond, depth)
                elif isinstance(node, ast.Return):
                    env["__return__"] = (
                        self.eval(node.value, env, module, cond, depth)
                        if node.value
                        else Value()
                    )
                    out.append((env, cond, True))
                    continue
                elif isinstance(node, ast.If):
                    test = self.eval(node.test, env, module, cond, depth)
                    for branch, truth in [(node.body, True), (node.orelse, False)]:
                        bc = None
                        if (
                            cond is not None
                            and test.expr is not None
                            and z3.is_bool(test.expr)
                        ):
                            bc = [*cond, test.expr if truth else z3.Not(test.expr)]
                            solver = z3.Solver()
                            solver.set(timeout=100)
                            solver.add(*bc)
                            if solver.check() == z3.unsat:
                                continue
                        elif self.symbolic_enabled:
                            self.warnings.add(
                                f"{module}:{node.lineno}: unmodeled branch guard"
                            )
                        out.extend(self.walk(branch, dict(env), module, bc, depth))
                    continue
                elif not isinstance(node, ast.Pass):
                    self.warnings.add(
                        f"{module}:{node.lineno}: unsupported statement {type(node).__name__}"
                    )
                    # Do not claim concrete witnesses downstream of unmodeled state changes.
                    cond = None
                out.append((env, cond, False))
            if len(out) > 64:
                self.warnings.add("symbolic state limit reached")
            states = out[:64]
        return states

    def run(self):
        initial_conditions = [] if self.symbolic_enabled else None
        for module, (path, tree) in self.modules.items():
            top_level_functions = {
                id(node)
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            # Structural sink inventory also covers statements outside the symbolic subset.
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.warnings.add(
                        f"{path}:{node.lineno}: class body/method semantics unsupported"
                    )
                elif (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and id(node) not in top_level_functions
                ):
                    self.warnings.add(
                        f"{path}:{node.lineno}: nested function/method semantics unsupported"
                    )
                if isinstance(node, ast.Call):
                    name = self.name(node.func, module, {})
                    if name in SINKS:
                        self.findings.append(
                            dict(
                                kind="sink",
                                path=path,
                                line=node.lineno,
                                detail=f"Call to {name}",
                                witness=None,
                            )
                        )
            self.walk(tree.body, {}, module, initial_conditions)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # A standalone entry point's `request` parameter follows the
                    # common HTTP-handler convention. This scoped assumption is
                    # explicit; other unknown objects are never guessed as sources.
                    # Actual interprocedural calls bind their supplied values above.
                    parameters = [
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    ]
                    env = {p.arg: Value() for p in parameters}
                    if "request" in env:
                        env["request"] = Value(symbol="request")
                        self.warnings.add(
                            f"{path}:{node.lineno}: standalone parameter 'request' assumed to be an HTTP request"
                        )
                    self.walk(node.body, env, module, initial_conditions)
        unique = {repr(f): f for f in self.findings}
        return list(unique.values()), sorted(self.warnings)


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
        "query": "medium",
        "execute": "medium",
    },
    "typescript": {
        "eval": "high",
        "Function": "high",
        "exec": "high",
        "execSync": "high",
        "spawn": "medium",
        "spawnSync": "medium",
        "query": "medium",
        "execute": "medium",
    },
    "tsx": {
        "eval": "high",
        "Function": "high",
        "exec": "high",
        "execSync": "high",
        "query": "medium",
        "execute": "medium",
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


def analyze_file(path: str, source: str, *, disabled=(), _include_flow=True) -> tuple[list[dict], list[str]]:
    """Analyze one non-Python source file. Returns findings and warnings."""
    disabled = set(disabled)
    language = language_for(path)
    if language is None:
        return [], [f"{path}: unsupported language for multilingual analysis"]
    if language not in _SUPPORTED:
        return [], [f"{path}: no multilingual sink rules for {language}"]
    if language == "python":
        return [], []
    if len(source.encode("utf-8")) > MAX_ANALYSIS_SOURCE_BYTES:
        return [], [f"{path}: source exceeds {MAX_ANALYSIS_SOURCE_BYTES} bytes; multilingual analysis skipped"]

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
    use_proximity = (
        "taint" not in disabled
        and language not in {"javascript", "typescript", "tsx", "php"}
    )

    findings: list[dict] = []
    calls = _collect_nodes(tree.root_node, call_types)
    functions = _collect_nodes(tree.root_node, function_types) if use_proximity else []

    # Walk parent links instead of scanning every function for every call.
    def enclosing(node):
        parent = node.parent
        while parent is not None:
            if parent.type in function_types:
                return parent
            parent = parent.parent
        return None

    # Per-function source presence cache
    fn_has_source: dict[tuple[int, int], bool] = {}
    for fn in functions:
        slice_text = data[fn.start_byte : fn.end_byte].decode("utf-8", errors="replace")
        fn_has_source[(fn.start_byte, fn.end_byte)] = any(p.search(slice_text) for p in source_pats)

    # File-level source fallback when there is no function scope (scripts).
    file_has_source = any(p.search(source) for p in source_pats) if use_proximity else False

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
        if not use_proximity:
            continue
        fn = enclosing(call)
        colocated = (
            fn_has_source.get((fn.start_byte, fn.end_byte), False)
            if fn is not None else file_has_source
        )
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

    if _include_flow and language in {"javascript", "typescript", "tsx", "php"}:
        flow_findings, flow_warnings = FlowAnalyzer(
            {path: source}, disabled=disabled
        ).run()
        findings.extend(flow_findings)
        warnings.extend(flow_warnings)
    unique = {repr(f): f for f in findings}
    return list(unique.values()), warnings


def analyze_non_python_files(files: dict[str, str], *, disabled=()) -> tuple[list[dict], list[str]]:
    """Analyze all non-Python files in ``files``."""
    findings: list[dict] = []
    warnings: list[str] = []
    analyzed = 0
    for path, source in files.items():
        language = language_for(path)
        if language in {None, "python"}:
            continue
        flow_language = language in {"javascript", "typescript", "tsx", "php"}
        file_findings, file_warnings = analyze_file(
            path, source,
            disabled=set(disabled) | {"taint"} if flow_language else disabled,
            _include_flow=False,
        )
        findings.extend(file_findings)
        warnings.extend(file_warnings)
        if language in _SUPPORTED:
            analyzed += 1
    flow_files = {path: source for path, source in files.items()
                  if language_for(path) in {"javascript", "typescript", "tsx", "php"}}
    if flow_files:
        flow_findings, flow_warnings = FlowAnalyzer(flow_files, disabled=disabled).run()
        findings.extend(flow_findings)
        warnings.extend(flow_warnings)
        warnings.append(
            f"{len(flow_files)} JavaScript/TypeScript/PHP candidate files: "
            "bounded AST flow applied where source size and syntax permit"
        )
    if analyzed and analyzed != len(flow_files):
        warnings.append(
            f"{analyzed - len(flow_files)} other non-Python files: "
            "sink/taint proximity heuristics applied"
        )
    return findings, warnings

class SemanticAnalyzer:
    """One entry point for all supported source languages."""

    def __init__(self, files: dict[str, str], *, disabled=()):
        self.files = files
        self.disabled = disabled

    def run(self) -> tuple[list[dict], list[str]]:
        python_findings, python_warnings = PythonAnalyzer(
            self.files, disabled=self.disabled
        ).run()
        other_findings, other_warnings = analyze_non_python_files(
            self.files, disabled=self.disabled
        )
        return python_findings + other_findings, python_warnings + other_warnings


# Bounded JavaScript/TypeScript/PHP AST flow.
FLOW_LANGS = {"javascript", "typescript", "tsx", "php"}
FLOW_CALL_TYPES = {"call_expression", "function_call_expression", "method_call_expression"}
FLOW_SINKS = {
    "eval", "Function", "exec", "execSync", "spawn", "spawnSync",
    "system", "passthru", "shell_exec", "popen", "proc_open", "assert",
    "mysqli_query", "mysql_query", "pg_query", "sqlite_query", "query",
    "execute", "executeQuery",
}
FLOW_HTML_CALL_SINKS = {"document.write", "res.send", "res.write"}
FLOW_SQL_SECOND_ARG = {"mysqli_query", "pg_query"}
FLOW_SOURCE_JS = re.compile(
    r"^(?:req|request)\.(?:body|query|params|cookies|headers)(?:\b|\[)|"
    r"^process\.argv(?:\b|\[)|^location\.(?:hash|search|href)\b|"
    r"^document\.cookie\b"
)
FLOW_SOURCE_PHP = re.compile(r"^\$_(?:GET|POST|REQUEST|COOKIE|SERVER|FILES)(?:\b|\[)")
FLOW_JS_IMPORT = re.compile(
    r"\bimport\s+(?:\{(?P<named>[^}]+)\}|(?P<default>[\w$]+))\s+"
    r"from\s*['\"](?P<path>\.[^'\"]+)['\"]"
)
FLOW_PHP_INCLUDE = re.compile(
    r"\b(?:require|require_once|include|include_once)\s*"
    r"(?:__DIR__\s*\.\s*)?['\"](?P<path>[^'\"]+\.php)['\"]"
)


def _flow_text(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def _flow_children(node):
    return node.named_children if node is not None else []


def _flow_field(node, name):
    return node.child_by_field_name(name) if node is not None else None


def _flow_module_path(origin: str, specifier: str, files: set[str]) -> str | None:
    if not specifier.startswith("."):
        return None
    parts = list(PurePosixPath(origin).parent.parts)
    for part in PurePosixPath(specifier).parts:
        if part == ".":
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    base = "/".join(parts)
    for candidate in (base, *(base + suffix for suffix in (".js", ".ts", ".tsx", ".php")),
                      *(base + "/index" + suffix for suffix in (".js", ".ts", ".tsx"))):
        if candidate in files:
            return candidate
    return None


@dataclass
class FlowValue:
    tainted: bool = False
    expr: object = None
    source: str | None = None


@dataclass
class FlowState:
    env: dict[str, FlowValue] = field(default_factory=dict)
    constraints: tuple = ()
    returned: bool = False
    result: FlowValue = field(default_factory=FlowValue)
    symbolic_ok: bool = True

    def copy(self) -> "FlowState":
        return FlowState(dict(self.env), self.constraints, self.returned,
                     self.result, self.symbolic_ok)


class FlowAnalyzer:
    def __init__(self, files: dict[str, str], *, disabled=(), max_steps=3000,
                 max_states=32, max_depth=3):
        self.disabled = set(disabled)
        candidates = {path: source for path, source in files.items()
                      if LANGUAGES.get(PurePosixPath(path).suffix.lower()) in FLOW_LANGS}
        self.files = {path: source for path, source in candidates.items()
                      if len(source.encode("utf-8")) <= MAX_ANALYSIS_SOURCE_BYTES}
        self.max_steps = max_steps
        self.max_states = max_states
        self.max_depth = max_depth
        self.steps = 0
        self.source_id = 0
        self.source_cache: dict[tuple[str, str], FlowValue] = {}
        self.findings: list[dict] = []
        self.warnings: set[str] = set()
        for path in candidates.keys() - self.files.keys():
            self.warnings.add(
                f"{path}: source exceeds {MAX_ANALYSIS_SOURCE_BYTES} bytes; traced flow skipped"
            )
        self.roots = {}
        self.functions = {}
        self.imports = {}
        self.includes = {}
        self._seen = set()
        for path, source in self.files.items():
            lang = LANGUAGES[PurePosixPath(path).suffix.lower()]
            root = get_parser(lang).parse(source.encode()).root_node
            if root.has_error:
                self.warnings.add(f"{path}: parse errors; traced flow skipped")
                continue
            self.roots[path] = root
            self.functions[path] = {}
            self.imports[path] = {}
            self.includes[path] = set()
            for node in _flow_children(root):
                candidates = [node]
                if node.type == "export_statement":
                    candidates.extend(_flow_children(node))
                for candidate in candidates:
                    if candidate.type == "function_declaration" or candidate.type == "function_definition":
                        name = _flow_text(_flow_field(candidate, "name"))
                        if name:
                            self.functions[path][name] = candidate
                if node.type == "import_statement":
                    match = FLOW_JS_IMPORT.search(_flow_text(node))
                    if match:
                        target = _flow_module_path(path, match["path"], set(self.files))
                        if target:
                            if match["named"]:
                                for item in match["named"].split(","):
                                    words = item.strip().split()
                                    if words:
                                        original = words[0]
                                        alias = words[-1]
                                        self.imports[path][alias] = (target, original)
                            elif match["default"]:
                                self.warnings.add(f"{path}: default import resolution not modeled")
                        else:
                            self.warnings.add(f"{path}: import target unavailable: {match['path']}")
                    else:
                        self.warnings.add(f"{path}: import form not modeled")
                if lang == "php" and "expression_statement" == node.type:
                    match = FLOW_PHP_INCLUDE.search(_flow_text(node))
                    if match:
                        specifier = match["path"]
                        if specifier.startswith("/") and "__DIR__" not in match.group(0):
                            self.warnings.add(f"{path}: absolute include not modeled")
                            continue
                        target = _flow_module_path(path, "./" + specifier.lstrip("/"), set(self.files))
                        if target:
                            self.includes[path].add(target)
                        else:
                            self.warnings.add(f"{path}: include target unavailable: {specifier}")
                    elif re.search(r"\b(?:require|include)(?:_once)?\b", _flow_text(node)):
                        self.warnings.add(f"{path}: dynamic include not modeled")

    def _source(self, path: str, node) -> FlowValue:
        key = (path, _flow_text(node))
        if key in self.source_cache:
            return self.source_cache[key]
        self.source_id += 1
        name = f"input_{self.source_id}"
        value = FlowValue(True, z3.String(name), f"{path}:{node.start_point[0] + 1}:{name}")
        self.source_cache[key] = value
        return value

    def _emit(self, kind: str, path: str, node, detail: str, witness=None):
        finding = dict(kind=kind, path=path, line=node.start_point[0] + 1,
                       detail=detail, witness=witness)
        key = (kind, path, finding["line"], detail)
        if key not in self._seen:
            self._seen.add(key)
            self.findings.append(finding)

    def _model(self, constraints, value: FlowValue):
        solver = z3.Solver()
        solver.set("timeout", 50)
        solver.add(*constraints)
        if solver.check() != z3.sat:
            return None
        if value.expr is None or not value.source:
            return None
        symbol = z3.String(value.source.rsplit(":", 1)[-1])
        result = solver.model().eval(symbol, model_completion=True)
        if z3.is_string_value(result):
            return {value.source: result.as_string()}
        return None

    def _feasible(self, constraints) -> bool:
        solver = z3.Solver()
        solver.set("timeout", 50)
        solver.add(*constraints)
        result = solver.check()
        if result == z3.unknown:
            self.warnings.add("symbolic solver timeout/unknown; path retained")
        return result != z3.unsat

    def _record_sink(self, path: str, node, name: str, value: FlowValue,
                     state: FlowState, crossed: bool, *, inventory=False) -> None:
        if inventory:
            self._emit("sink", path, node, f"Call or write to {name}")
        if not value.tainted or "taint" in self.disabled:
            return
        detail = f"Modeled source-to-sink flow into {name} from {value.source}"
        self._emit("taint", path, node, detail)
        if crossed and "cross_file" not in self.disabled:
            self._emit("cross_file", path, node, detail)
        if "symbolic" not in self.disabled and state.symbolic_ok and value.expr is not None:
            witness = self._model(state.constraints, value)
            if witness is not None:
                self._emit("symbolic", path, node,
                           f"Feasible modeled path to {name}", witness)

    def _expr(self, path: str, node, state: FlowState, depth: int, crossed: bool) -> FlowValue:
        if node is None:
            return FlowValue()
        kind = node.type
        raw = _flow_text(node)
        lang = LANGUAGES[PurePosixPath(path).suffix.lower()]
        if kind in {"identifier", "variable_name"}:
            if lang == "php" and FLOW_SOURCE_PHP.match(raw):
                return self._source(path, node)
            return state.env.get(raw, FlowValue())
        if kind in {"member_expression", "subscript_expression"}:
            if (FLOW_SOURCE_PHP if lang == "php" else FLOW_SOURCE_JS).match(raw):
                return self._source(path, node)
            values = [self._expr(path, child, state, depth, crossed) for child in _flow_children(node)
                      if child.type not in {"property_identifier", "string", "encapsed_string"}]
            return next((value for value in values if value.tainted), FlowValue())
        if kind in {"string", "encapsed_string"}:
            if kind == "encapsed_string" and any(
                child.type == "variable_name" for child in _flow_children(node)
            ):
                values = [self._expr(path, child, state, depth, crossed)
                          for child in _flow_children(node)]
                tainted = next((value for value in values if value.tainted), None)
                return FlowValue(True, None, tainted.source) if tainted else FlowValue()
            if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
                return FlowValue(False, z3.StringVal(raw[1:-1]))
            return FlowValue()
        if kind in FLOW_CALL_TYPES:
            return self._call(path, node, state, depth, crossed)
        if kind == "print_intrinsic":
            value = self._expr(path, _flow_children(node)[0] if _flow_children(node) else None,
                               state, depth, crossed)
            self._record_sink(path, node, "print", value, state, crossed,
                              inventory=True)
            return FlowValue()
        if kind in {"parenthesized_expression", "argument"}:
            children = _flow_children(node)
            return self._expr(path, children[0], state, depth, crossed) if children else FlowValue()
        if kind == "binary_expression":
            left, right = _flow_field(node, "left"), _flow_field(node, "right")
            a = self._expr(path, left, state, depth, crossed)
            b = self._expr(path, right, state, depth, crossed)
            source = a.source if a.tainted else b.source
            operator = _flow_text(node)[len(_flow_text(left)):-len(_flow_text(right))].strip() if left and right else ""
            if operator in {"+", "."} and a.expr is not None and b.expr is not None:
                try:
                    return FlowValue(a.tainted or b.tainted, z3.Concat(a.expr, b.expr), source)
                except z3.Z3Exception:
                    pass
            return FlowValue(a.tainted or b.tainted, None, source)
        if kind in {"template_string", "encapsed_string_fragment", "concatenation_expression"}:
            parts = [self._expr(path, child, state, depth, crossed) for child in _flow_children(node)]
            tainted = next((part for part in parts if part.tainted), None)
            if tainted:
                return FlowValue(True, None, tainted.source)
            return FlowValue()
        if kind == "assignment_expression":
            right = self._expr(path, _flow_field(node, "right"), state, depth, crossed)
            left = _flow_field(node, "left")
            if left is not None and left.type in {"identifier", "variable_name"}:
                state.env[_flow_text(left)] = right
            elif left is not None and re.search(r"(?:\.innerHTML|\[['\"]innerHTML['\"]\])$", _flow_text(left)):
                self._record_sink(path, node, "innerHTML", right, state, crossed,
                                  inventory=True)
            else:
                self.warnings.add(f"{path}: unsupported assignment target")
            return right
        values = [self._expr(path, child, state, depth, crossed) for child in _flow_children(node)]
        tainted = next((value for value in values if value.tainted), None)
        if tainted:
            self.warnings.add(f"{path}: approximate flow through {kind}")
            return tainted
        return FlowValue()

    def _call(self, path: str, node, state: FlowState, depth: int, crossed: bool) -> FlowValue:
        function = _flow_field(node, "function")
        full_name = _flow_text(function)
        name = full_name.split(".")[-1].split("->")[-1]
        arguments = _flow_field(node, "arguments")
        args = [self._expr(path, arg, state, depth, crossed) for arg in _flow_children(arguments)]
        if name in FLOW_SINKS or full_name in FLOW_HTML_CALL_SINKS:
            index = 1 if name in FLOW_SQL_SECOND_ARG and len(args) > 1 else 0
            value = args[index] if len(args) > index else FlowValue()
            self._record_sink(path, node, full_name, value, state, crossed,
                              inventory=full_name in FLOW_HTML_CALL_SINKS)
            return FlowValue()
        target_path, target_name = path, name
        if name in self.imports.get(path, {}):
            target_path, target_name = self.imports[path][name]
        elif name not in self.functions.get(path, {}):
            linked = [other for other in self.includes.get(path, ())
                      if name in self.functions.get(other, {})]
            if len(linked) == 1:
                target_path = linked[0]
        if target_path != path and "cross_file" in self.disabled:
            return FlowValue()
        fn = self.functions.get(target_path, {}).get(target_name)
        if fn is None:
            if any(arg.tainted for arg in args):
                self.warnings.add(f"{path}: tainted call to unresolved function {name}")
            return FlowValue()
        if depth >= self.max_depth:
            self.warnings.add(f"{path}: call-depth limit reached")
            return FlowValue()
        parameters = _flow_field(fn, "parameters")
        names = []
        for param in _flow_children(parameters):
            name_node = _flow_field(param, "name") or param
            names.append(_flow_text(name_node))
        callee = FlowState(dict(zip(names, args)), state.constraints,
                       symbolic_ok=state.symbolic_ok)
        body = _flow_field(fn, "body")
        outcomes = self._statements(target_path, _flow_children(body), [callee],
                                    depth + 1, crossed or target_path != path)
        returned = [outcome.result for outcome in outcomes if outcome.returned]
        return next((value for value in returned if value.tainted),
                    returned[0] if returned else FlowValue())

    def _condition(self, path: str, node, state: FlowState, depth: int, crossed: bool):
        if "symbolic" in self.disabled or "taint" in self.disabled:
            return None
        if node is None:
            return None
        if node.type == "parenthesized_expression" and _flow_children(node):
            node = _flow_children(node)[0]
        if node.type != "binary_expression":
            self.warnings.add(f"{path}: unsupported branch condition")
            return None
        left, right = _flow_field(node, "left"), _flow_field(node, "right")
        operator = _flow_text(node)[len(_flow_text(left)):-len(_flow_text(right))].strip() if left and right else ""
        if operator not in {"==", "===", "!=", "!=="}:
            self.warnings.add(f"{path}: unsupported branch operator {operator}")
            return None
        a = self._expr(path, left, state, depth, crossed)
        b = self._expr(path, right, state, depth, crossed)
        if a.expr is None or b.expr is None:
            return None
        try:
            eq = a.expr == b.expr
            return z3.Not(eq) if operator in {"!=", "!=="} else eq
        except z3.Z3Exception:
            self.warnings.add(f"{path}: unsupported branch value types")
            return None

    def _statements(self, path: str, nodes, states: list[FlowState], depth: int,
                    crossed: bool) -> list[FlowState]:
        for node in nodes:
            next_states = []
            for state in states:
                if state.returned:
                    next_states.append(state)
                    continue
                self.steps += 1
                if self.steps > self.max_steps:
                    self.warnings.add("multilingual flow step limit reached")
                    return states
                kind = node.type
                if kind in {"lexical_declaration", "variable_declaration"}:
                    for declarator in _flow_children(node):
                        if declarator.type == "variable_declarator":
                            name = _flow_field(declarator, "name")
                            value = self._expr(path, _flow_field(declarator, "value"), state,
                                               depth, crossed)
                            if name is not None and name.type == "identifier":
                                state.env[_flow_text(name)] = value
                            else:
                                self.warnings.add(f"{path}: destructuring/type pattern not modeled")
                    next_states.append(state)
                elif kind == "if_statement":
                    condition = self._condition(path, _flow_field(node, "condition"), state,
                                                depth, crossed)
                    then = _flow_field(node, "consequence") or _flow_field(node, "body")
                    otherwise = _flow_field(node, "alternative")
                    if otherwise is None:
                        otherwise = next((child for child in _flow_children(node)
                                          if child.type == "else_clause"), None)
                    for truth, branch in ((True, then), (False, otherwise)):
                        fork = state.copy()
                        if condition is not None:
                            fork.constraints += (condition if truth else z3.Not(condition),)
                            if not self._feasible(fork.constraints):
                                continue
                        else:
                            fork.symbolic_ok = False
                        body = _flow_children(branch) if branch is not None else []
                        if branch is not None and branch.type == "else_clause":
                            body = _flow_children(body[0]) if body else []
                        next_states.extend(self._statements(path, body, [fork], depth, crossed))
                elif kind == "return_statement":
                    state.result = self._expr(path, _flow_children(node)[0] if _flow_children(node)
                                              else None, state, depth, crossed)
                    state.returned = True
                    next_states.append(state)
                elif kind in {"statement_block", "compound_statement"}:
                    next_states.extend(self._statements(path, _flow_children(node), [state],
                                                        depth, crossed))
                elif kind == "expression_statement":
                    for child in _flow_children(node):
                        self._expr(path, child, state, depth, crossed)
                    next_states.append(state)
                elif kind == "echo_statement":
                    for child in _flow_children(node):
                        value = self._expr(path, child, state, depth, crossed)
                        self._record_sink(path, node, "echo", value, state, crossed,
                                          inventory=True)
                    next_states.append(state)
                elif kind in {"import_statement", "php_tag", "export_statement"}:
                    next_states.append(state)
                else:
                    self.warnings.add(f"{path}: {kind} flow not modeled")
                    next_states.append(state)
            states = next_states[:self.max_states]
            if len(next_states) > self.max_states:
                self.warnings.add("multilingual flow state limit reached")
        return states

    def run(self) -> tuple[list[dict], list[str]]:
        for path, root in self.roots.items():
            top_level = [node for node in _flow_children(root)
                         if node.type not in {"function_declaration", "function_definition"}]
            self._statements(path, top_level, [FlowState()], 0, False)
            for fn in self.functions[path].values():
                self._statements(path, _flow_children(_flow_field(fn, "body")), [FlowState()], 0, False)
        return self.findings, sorted(self.warnings)
