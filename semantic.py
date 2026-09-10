"""Bounded Python symbolic interpreter with import-aware calls and taint.

No submitted code is executed. Z3 produces witnesses for supported paths.
Unsupported operations, loops, dynamic dispatch, and exhausted bounds are recorded.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import PurePosixPath
import z3

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
