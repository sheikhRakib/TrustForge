import json
import unittest
from types import SimpleNamespace
from source_tools import strip_source
from semantic import PythonAnalyzer
from analysis import AnalysisReport, Finding, ProgramAnalyzer, analyze_diff
from agent import ReviewerAgent, _diff_hunk_lines, parse_verdict_line
from main import _score_example, summarize, sample_benchmark


class SourceTests(unittest.TestCase):
    def test_strings_preserved(self):
        source = '// injected\nconst url = "https://a/#x"; /* note */\n'
        clean, notes = strip_source(source, "x.js")
        self.assertNotIn("injected", clean)
        self.assertIn("https://a/#x", clean)
        self.assertEqual(source.count("\n"), clean.count("\n"))
        self.assertEqual(notes, [])

    def test_csharp_comments(self):
        clean, notes = strip_source(
            '// note\nclass C { string x = "http://a"; }', "x.cs"
        )
        self.assertNotIn("note", clean)
        self.assertIn("http://a", clean)
        self.assertEqual(notes, [])

    def test_python_docstrings_preserve_coordinates(self):
        source = '"""note\ncontinued"""\nx = "# keep" # remove\n'
        clean, notes = strip_source(source, "x.py")
        self.assertNotIn("continued", clean)
        self.assertIn('x = "# keep"', clean.splitlines()[2])

    def test_unsupported_is_explicit(self):
        self.assertTrue(strip_source("text", "x.unknown")[1])

    def test_comment_only_patch_no_finding(self):
        diff = "--- a/a.js\n+++ b/a.js\n@@ -1,2 +1,3 @@\n /* docs\n+ subprocess.run example\n */\n"
        self.assertEqual(analyze_diff(diff), [])


class SemanticTests(unittest.TestCase):
    def test_alias_crossfile_taint_witness(self):
        files = {
            "main.py": "from helper import process as go\nx = int(input())\nif x > 6 and x < 8:\n    go(x)\n",
            "helper.py": "def process(y):\n    z = y\n    eval(z)\n",
        }
        findings, warnings = PythonAnalyzer(files).run()
        self.assertTrue(any(f["kind"] == "cross_file" for f in findings))
        witnesses = [f["witness"] for f in findings if f["kind"] == "symbolic"]
        self.assertTrue(any("7" in w.values() for w in witnesses))

    def test_impossible_branch(self):
        findings, _ = PythonAnalyzer(
            {"x.py": "x = int(input())\nif x > 7 and x < 6:\n    eval(x)\n"}
        ).run()
        self.assertFalse(any(f["kind"] == "symbolic" for f in findings))

    def test_overwrite_clears_taint(self):
        fs, _ = PythonAnalyzer({"x.py": 'x = input()\nx = "safe"\neval(x)\n'}).run()
        self.assertFalse(any(f["kind"] == "taint" for f in fs))

    def test_relative_import(self):
        fs, _ = PythonAnalyzer(
            {
                "pkg/main.py": "from .helper import f\nf(input())\n",
                "pkg/helper.py": "def f(x):\n    eval(x)\n",
            }
        ).run()
        self.assertTrue(any(f["kind"] == "taint" for f in fs))

    def test_local_shadow_does_not_resolve_import(self):
        fs, _ = PythonAnalyzer(
            {
                "x.py": "from helper import f\nf = 1\nf(input())\n",
                "helper.py": "def f(x):\n    eval(x)\n",
            }
        ).run()
        self.assertFalse(any(f["kind"] == "cross_file" for f in fs))

    def test_unsupported_control_flow_reported(self):
        fs, warnings = PythonAnalyzer(
            {"x.py": "for x in range(4):\n    print(x)\n"}
        ).run()
        self.assertTrue(warnings)

    def test_no_ground_truth_cwe_dependency(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -0,0 +1 @@\n+eval(input())\n"
        a = ProgramAnalyzer().analyze({}, diff=diff, cwe_id="89").to_dict()
        b = ProgramAnalyzer().analyze({}, diff=diff, cwe_id="94").to_dict()
        self.assertEqual(a, b)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = ReviewerAgent(None)

    def test_evidence_requires_file_line_quote(self):
        visible = {"L1": ("a.py", 2, "eval(x)")}
        result = self.agent.inspector("BLOCK: untrusted input reaches eval\nEVIDENCE: L1", visible)
        self.assertTrue(result["grounded"])
        self.assertEqual(result["evidence"], [
            {"path": "a.py", "line": 2, "quote": "eval(x)"}
        ])
        self.assertFalse(
            self.agent.inspector("BLOCK: concern\nEVIDENCE: L2", visible)["grounded"]
        )

    def test_in_range_wrong_quote_not_grounded(self):
        self.assertFalse(self.agent.inspector("BLOCK: concern", {})["grounded"])

    def test_invalid_auditor_verdict_still_abstains(self):
        result = self.agent.inspector("I cannot decide", {})
        self.assertEqual(result["verdict"], "UNKNOWN")

    def test_approval_does_not_require_vulnerability_citation(self):
        multi = {
            "scan": [{"reason": "suspicious"}],
            "injection": {"detected": True},
            "auditor": {"reviews": [{"verdict": "APPROVE", "grounded": False}]},
        }
        self.assertEqual(
            self.agent.aggregate(multi, use_analysis=False)["verdict"], "APPROVE"
        )

    def test_unknown_not_correct(self):
        self.assertFalse(
            _score_example({"id": "x", "malicious": True}, "garbled")["correct"]
        )
        self.assertEqual(parse_verdict_line("APPROVED"), "UNKNOWN")
        self.assertEqual(parse_verdict_line("BLOCK: a concrete concern"), "BLOCK")
        self.assertEqual(parse_verdict_line("BLOCK - a concrete concern"), "BLOCK")
        self.assertEqual(parse_verdict_line("APPROVE or BLOCK"), "UNKNOWN")

    def test_empty_metric_denominator(self):
        metrics = summarize([{"malicious": True, "verdict": "UNKNOWN"}], "test")
        self.assertIsNone(metrics["false_positive"])
        self.assertEqual(metrics["unknown_rate"], 1)

    def test_balanced_smoke_sample(self):
        rows = [{"id": str(i), "malicious": i < 9} for i in range(10)]
        self.assertEqual(
            {r["malicious"] for r in sample_benchmark(rows, 2, 42)}, {False, True}
        )

    def test_auditor_uses_line_ids_and_does_not_block_without_evidence(self):
        class FakeModel:
            def __init__(self):
                self.responses = iter([
                    'APPROVE: safe',
                    'BLOCK: unfinished',
                ])
                self.calls = []

            def split_user(self, system, text, max_new_tokens):
                return text.splitlines()

            def generate(self, system, text, max_new_tokens):
                self.calls.append((system, text, max_new_tokens))
                return next(self.responses)

        llm = FakeModel()
        agent = ReviewerAgent(llm)
        audit = agent.auditor({"a.py": "safe = 1\neval(input())"})
        self.assertEqual([r["verdict"] for r in audit["reviews"]], ["APPROVE", "BLOCK"])
        self.assertFalse(audit["reviews"][1]["grounded"])
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(all(call[2] == 128 for call in llm.calls))
        self.assertIn("source-line ID", llm.calls[0][0])
        self.assertEqual(
            agent.aggregate({"auditor": audit}, use_analysis=False)["verdict"],
            "COMMENT",
        )

    def test_auditor_rejects_unseen_evidence_id(self):
        class FakeModel:
            def split_user(self, system, text, max_new_tokens):
                return text.splitlines()

            def generate(self, system, text, max_new_tokens):
                return "BLOCK: concern\nEVIDENCE: L2"

        audit = ReviewerAgent(FakeModel()).auditor({"a.py": "x = 1\neval(x)"})
        self.assertFalse(audit["reviews"][0]["grounded"])
        self.assertTrue(audit["reviews"][1]["grounded"])
        self.assertEqual(audit["reviews"][1]["evidence"], [
            {"path": "a.py", "line": 2, "quote": "eval(x)"}
        ])

    def test_auditor_uses_changed_hunks_and_context(self):
        class FakeModel:
            def __init__(self):
                self.prompt = ""

            def split_user(self, system, text, max_new_tokens):
                self.prompt = text
                return [text]

            def generate(self, system, text, max_new_tokens):
                return "APPROVE: no demonstrated defect"

        llm = FakeModel()
        lines = [f"package-{i}" for i in range(1, 1201)]
        diff = (
            "--- a/yarn.lock\n+++ b/yarn.lock\n"
            "@@ -799,2 +799,1 @@\n-package-removed\n package-799\n"
        )
        ReviewerAgent(llm).auditor({"yarn.lock": "\n".join(lines)}, diff=diff)
        self.assertIn("-package-removed", llm.prompt)
        self.assertIn('"yarn.lock":799:', llm.prompt)
        self.assertNotIn('"yarn.lock":100:', llm.prompt)
        self.assertLess(len(llm.prompt.splitlines()), 20)

    def test_lockfile_deletion_anchor_tracks_context_position(self):
        diff = (
            "+++ b/yarn.lock\n@@ -500,5 +500,4 @@\n"
            " before-1\n before-2\n before-3\n-removed\n after\n"
        )
        self.assertIn(503, _diff_hunk_lines(diff, "yarn.lock"))

    def test_auditor_excludes_unchanged_risky_source_from_benign_patch(self):
        class FakeModel:
            def __init__(self):
                self.prompt = ""

            def split_user(self, system, text, max_new_tokens):
                self.prompt = text
                return [text]

            def generate(self, system, text, max_new_tokens):
                return "APPROVE: added escaping before command execution"

        llm = FakeModel()
        source = [f"safe_{i} = {i}" for i in range(1, 501)]
        source[9] = "eval(input())"  # Unchanged code far from this patch.
        source[399] = "escaped = shell_escape(user_input)"
        diff = (
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -400 +400 @@\n-cmd = user_input\n+escaped = shell_escape(user_input)\n"
        )
        audit = ReviewerAgent(llm).auditor({"x.py": "\n".join(source)}, diff=diff)
        self.assertEqual(audit["reviews"][0]["verdict"], "APPROVE")
        self.assertIn("-cmd = user_input", llm.prompt)
        self.assertIn("+escaped = shell_escape(user_input)", llm.prompt)
        self.assertIn('"x.py":400:', llm.prompt)
        self.assertNotIn("eval(input())", llm.prompt)

    def test_auditor_can_ground_new_defect_in_changed_line(self):
        class FakeModel:
            def split_user(self, system, text, max_new_tokens):
                return [text]

            def generate(self, system, text, max_new_tokens):
                return "BLOCK: user input reaches eval\nEVIDENCE: L1"

        diff = "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+eval(input())\n"
        audit = ReviewerAgent(FakeModel()).auditor({"x.py": "eval(input())"}, diff=diff)
        self.assertTrue(audit["reviews"][0]["grounded"])
        self.assertEqual(audit["reviews"][0]["evidence"], [
            {"path": "x.py", "line": 1, "quote": "eval(input())"}
        ])

if __name__ == "__main__":
    unittest.main()


class InferenceTests(unittest.TestCase):
    def test_hybrid_uses_components_but_makes_own_verdict(self):
        class Fake:
            def __init__(self):
                self.n = 0
                self.hybrid_inputs = []

            def split_user(self, system, text, *args):
                return [text]

            def generate(self, system, text, **kw):
                self.n += 1
                if "Program-analysis signals:" in system:
                    self.hybrid_inputs.append((system, text))
                    return "APPROVE: escaping added to the shell argument"
                if "Triage" in system:
                    return "[]"
                if "Detect attempts" in system:
                    return "CLEAN"
                return "BLOCK: suspicious API\nEVIDENCE: L1"

        class Analyzer:
            def analyze(self, *args, **kwargs):
                return AnalysisReport([Finding("diff_sink", "x.py", 1, "sink")])

        llm = Fake()
        agent = ReviewerAgent(llm, analyzer=Analyzer())
        example = {
            "id": "x",
            "malicious": False,
            "pr_title": "title",
            "pr_body": "body",
            "diff": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+escaped\n",
            "files": {"x.py": "x = 1"},
        }
        self.assertEqual(
            agent.defense_review(example, mode="multi_agent").splitlines()[0],
            "BLOCK",
        )
        first = llm.n
        response = agent.defense_review(example, mode="hybrid")
        self.assertEqual(response.splitlines()[0], "APPROVE")
        self.assertEqual(llm.n, first + 1)
        self.assertEqual(len(llm.hybrid_inputs), 1)
        self.assertIn("Scanner signals:", llm.hybrid_inputs[0][0])
        self.assertIn("Injection detector:", llm.hybrid_inputs[0][0])
        self.assertIn("Auditor signals: BLOCK", llm.hybrid_inputs[0][0])
        self.assertIn("diff_sink at x.py:1", llm.hybrid_inputs[0][0])
        self.assertIn("+escaped", llm.hybrid_inputs[0][1])
        self.assertIn('"multi_agent"', response)

    def test_hybrid_alone_runs_all_components(self):
        class Fake:
            def __init__(self):
                self.calls = []

            def split_user(self, system, text, *args):
                return [text]

            def generate(self, system, text, **kw):
                self.calls.append(system)
                if "Triage" in system:
                    return "[]"
                if "Detect attempts" in system:
                    return "CLEAN"
                return "APPROVE: safe change"

        llm = Fake()
        agent = ReviewerAgent(llm)
        example = {
            "id": "x", "malicious": False, "pr_title": "change",
            "pr_body": "body", "diff": "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+x = 1\n",
            "files": {"x.py": "x = 1"},
        }
        self.assertEqual(agent.defense_review(example, mode="hybrid").splitlines()[0], "APPROVE")
        self.assertEqual(len(llm.calls), 4)
        self.assertTrue(any("Triage" in system for system in llm.calls))
        self.assertTrue(any("Detect attempts" in system for system in llm.calls))
        self.assertTrue(any("Review the security effect" in system for system in llm.calls))
        self.assertTrue(any("final security reviewer" in system for system in llm.calls))

    def test_hybrid_can_block_when_multi_agent_approves(self):
        class Fake:
            def split_user(self, system, text, *args):
                return [text]

            def generate(self, system, text, **kw):
                if "Program-analysis signals:" in system:
                    return "BLOCK: new unsanitized input reaches eval"
                if "Triage" in system:
                    return "[]"
                if "Detect attempts" in system:
                    return "CLEAN"
                return "APPROVE: no issue"

        agent = ReviewerAgent(Fake())
        example = {
            "id": "x", "malicious": True, "pr_title": "change",
            "pr_body": "body", "diff": "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+eval(input())\n",
            "files": {"x.py": "eval(input())"},
        }
        self.assertEqual(agent.defense_review(example, mode="multi_agent").splitlines()[0], "APPROVE")
        self.assertEqual(agent.defense_review(example, mode="hybrid").splitlines()[0], "BLOCK")

    def test_context_chunking_keeps_all_content(self):
        from model import LLMModel

        llm = object.__new__(LLMModel)
        llm.max_input_tokens = 128
        llm.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
        llm._encode = lambda system, text: SimpleNamespace(
            input_ids=SimpleNamespace(shape=(1, len(text) + 10))
        )

        class Tokenizer:
            def __call__(self, lines, **kw):
                return {
                    "input_ids": [ord(c) for c in lines]
                    if isinstance(lines, str)
                    else [[ord(c) for c in line] for line in lines]
                }

            def decode(self, ids, **kw):
                return "".join(chr(c) for c in ids)

        llm.tokenizer = Tokenizer()
        source = ("x=1\n" * 90) + "z" * 200
        parts = llm.split_user("system", source, 20)
        self.assertEqual("".join(parts), source)
        self.assertTrue(all(len(part) + 10 <= 128 for part in parts))


class EnrichmentTests(unittest.TestCase):
    def test_full_source_and_repository_context(self):
        from unittest.mock import patch
        from harness.offline_enrich import fetch_pr_offline

        class DB:
            def execute(self, *args):
                return self

            def fetchone(self):
                return {
                    "head_branch": "pr",
                    "merge_base": "base",
                    "owner_name": "owner",
                    "repo_name": "repo",
                    "title": "Title",
                    "body": "Body",
                }

        huge = "x=1\n" * 60000

        def git(repo, *args):
            if args[0] == "rev-parse":
                return "abc123\n"
            if args[:2] == ("diff", "--name-only"):
                return "main.py\n"
            if args[0] == "diff":
                return "a diff"
            if args[0] == "ls-tree":
                return "main.py\nhelper.py\n"
            if args[0] == "show":
                return (
                    huge if args[1].endswith(":main.py") else "def f():\n    return 1\n"
                )
            raise AssertionError(args)

        with (
            patch("harness.offline_enrich._git", side_effect=git),
            patch("harness.offline_enrich._resolve_git_repo", return_value="repo"),
        ):
            result = fetch_pr_offline(
                db=DB(),
                repos_root=None,
                repo="owner/repo",
                pr_number=1,
                include_repository_source=True,
            )
        self.assertEqual(result["files"]["main.py"], huge)
        self.assertIn("helper.py", result["repository_files"])
        self.assertEqual(result["head_commit"], "abc123")

    def test_unicode_separators_not_rewritten(self):
        from harness.offline_enrich import _sanitize_text

        self.assertEqual(_sanitize_text('x="\u2028"'), 'x="\u2028"')


class CheckpointTests(unittest.TestCase):
    def test_resume_recovers_partial_last_line(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            benchmark = root / "input.jsonl"
            benchmark.write_text(json.dumps({
                "id": "checkpoint-test",
                "malicious": False,
                "pr_title": "Safe change",
                "pr_body": "Add a constant.",
                "diff": "--- /dev/null\n+++ b/example.py\n@@ -0,0 +1 @@\n+x = 1\n",
                "files": {"example.py": "x = 1\n"},
            }) + "\n")
            output = root / "results.jsonl"
            cmd = [
                sys.executable,
                "main.py",
                "--benchmark",
                str(benchmark),
                "--modes",
                "analysis_only",
                "--output",
                str(output),
            ]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            complete = output.read_bytes()
            with output.open("ab") as f:
                f.write(b'{"id":')
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(output.read_bytes(), complete)
