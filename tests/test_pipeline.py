import json
import unittest
from types import SimpleNamespace
from source_tools import strip_source
from semantic import PythonAnalyzer
from analysis import ProgramAnalyzer, analyze_diff
from agent import ReviewerAgent, parse_verdict_line
from main import _score_example, summarize, sample_benchmark
from augment import variants, semantic_fixtures


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
        source = {"a.py": "x = 1\neval(x)"}
        response = {
            "verdict": "BLOCK",
            "reason": "example",
            "evidence": [{"path": "a.py", "line": 2, "quote": "eval(x)"}],
        }
        self.assertTrue(self.agent.inspector(json.dumps(response), source)["grounded"])
        response["evidence"][0]["path"] = "other.py"
        self.assertFalse(self.agent.inspector(json.dumps(response), source)["grounded"])

    def test_in_range_wrong_quote_not_grounded(self):
        response = {
            "verdict": "BLOCK",
            "evidence": [{"path": "a.py", "line": 1, "quote": "eval"}],
        }
        self.assertFalse(
            self.agent.inspector(json.dumps(response), {"a.py": "x=1"})["grounded"]
        )

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

    def test_augmentation_preserves_code(self):
        row = {
            "id": "x",
            "malicious": True,
            "pr_body": "body",
            "files": {"x.py": "pass"},
            "diff": "diff",
        }
        outputs = list(variants(row))
        self.assertEqual(len(outputs), 4)
        self.assertTrue(
            all(
                r["files"] == row["files"] and r["diff"] == row["diff"] for r in outputs
            )
        )
        self.assertEqual(row["pr_body"], "body")

    def test_semantic_fixture_sources_separate(self):
        rows = list(semantic_fixtures())
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["source"] == "synthetic" for r in rows))


if __name__ == "__main__":
    unittest.main()


class InferenceTests(unittest.TestCase):
    def test_multi_agent_hybrid_reuse_calls(self):
        class Fake:
            def __init__(self):
                self.n = 0

            def split_user(self, system, text, *args):
                return [text]

            def generate(self, system, text, **kw):
                self.n += 1
                if "Triage" in system:
                    return "[]"
                if "Detect attempts" in system:
                    return "CLEAN"
                return '{"verdict":"APPROVE","reason":"safe","evidence":[]}'

        llm = Fake()
        agent = ReviewerAgent(llm)
        example = {
            "id": "x",
            "malicious": False,
            "pr_title": "title",
            "pr_body": "body",
            "diff": "diff",
            "files": {"x.py": "x = 1"},
        }
        agent.defense_review(example, mode="multi_agent")
        first = llm.n
        agent.defense_review(example, mode="hybrid")
        self.assertEqual(llm.n, first)

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
    def test_resume_recovers_partial_last_line_and_rejects_changed_data(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            benchmark = root / "input.jsonl"
            benchmark.write_text(
                "".join(json.dumps(r) + "\n" for r in semantic_fixtures())
            )
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
            rows = [json.loads(line) for line in benchmark.read_text().splitlines()]
            rows[0]["pr_body"] += " changed"
            benchmark.write_text("".join(json.dumps(r) + "\n" for r in rows))
            third = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(third.returncode, 0)
            self.assertIn("manifest differs", third.stderr)
