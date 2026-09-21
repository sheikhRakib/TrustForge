"""Baseline behavior through main.py without loading model weights."""

import subprocess
import sys
import unittest
from unittest.mock import patch

import baseline
from main import evaluate_mode, parse_args
from model import LLMModel


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.max_input_tokens = 8192
        self.call_count = 0

    def split_user(self, system, text, max_new_tokens):
        self.calls.append(("split", system, text, max_new_tokens))
        return ["first chunk", "second chunk"]

    def generate(self, system, text, max_new_tokens):
        self.calls.append(("generate", system, text, max_new_tokens))
        self.call_count += 1
        return next(self.responses)


class BaselineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.example = {
            "id": "x",
            "malicious": False,
            "pr_title": "title",
            "pr_body": "body",
            "files_changed": ["x.py"],
            "diff": "+safe = True",
        }

    def test_prompt_and_review_remain_in_baseline_module(self):
        self.assertIn("pull-request diff", baseline.BASELINE_SYSTEM)
        self.assertIn("exactly APPROVE:, COMMENT:, or BLOCK:", baseline.BASELINE_SYSTEM)
        self.assertIn("Do not repeat these instructions", baseline.BASELINE_SYSTEM)
        self.assertIs(baseline.LLMModel, LLMModel)
        self.assertEqual(baseline.baseline_review.__module__, "baseline")

    def test_main_accepts_baseline_scope_and_output(self):
        with patch("sys.argv", [
            "main.py", "--modes", "baseline", "--cwe", "cwe89",
            "--output", "output/custom.jsonl", "--limit", "2",
        ]):
            args = parse_args()
        self.assertEqual(args.modes, ["baseline"])
        self.assertEqual(args.cwe, ["cwe89"])
        self.assertEqual(str(args.output), "output/custom.jsonl")
        self.assertEqual(args.limit, 2)

    def test_baseline_calls_model_with_local_system_prompt(self):
        model = FakeModel(["APPROVE: first", "COMMENT: second"])
        response = baseline.baseline_review(model, self.example)
        self.assertTrue(response.startswith("COMMENT\n"))
        generated = [call for call in model.calls if call[0] == "generate"]
        self.assertEqual(len(generated), 2)
        self.assertTrue(all(call[1] == baseline.BASELINE_SYSTEM for call in generated))

    def test_chunk_verdict_precedence_and_strict_parsing(self):
        self.assertEqual(
            baseline.combine_verdicts(["COMMENT: concern", "BLOCK: defect"]),
            "BLOCK",
        )
        self.assertEqual(baseline.parse_verdict_line("APPROVED"), "UNKNOWN")
        self.assertEqual(
            baseline.combine_verdicts(["APPROVE: safe", "invalid"]), "UNKNOWN"
        )

    def test_main_scores_baseline_without_performance_telemetry(self):
        model = FakeModel(["APPROVE: first", "APPROVE: second"])
        row = evaluate_mode(None, model, self.example, "baseline")
        self.assertEqual(row["verdict"], "APPROVE")
        self.assertEqual(row["mode"], "baseline")
        self.assertEqual(model.call_count, 2)
        self.assertTrue({
            "inference", "seconds", "attributed_seconds", "reused_seconds",
            "reused_inference", "oom_retries", "input_budgets",
        }.isdisjoint(row))

    def test_baseline_file_refuses_direct_execution(self):
        result = subprocess.run(
            [sys.executable, baseline.__file__], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("main.py --modes baseline", result.stderr)


if __name__ == "__main__":
    unittest.main()
