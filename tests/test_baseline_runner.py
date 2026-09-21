"""Standalone baseline behavior without loading model weights."""

from pathlib import Path
import unittest

import baseline
from model import LLMModel


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def split_user(self, system, text, max_new_tokens):
        self.calls.append(("split", system, text, max_new_tokens))
        return ["first chunk", "second chunk"]

    def generate(self, system, text, max_new_tokens):
        self.calls.append(("generate", system, text, max_new_tokens))
        return next(self.responses)


class BaselineRunnerTests(unittest.TestCase):
    def test_prompt_model_and_review_are_defined_in_standalone_module(self):
        self.assertIn("pull-request diff", baseline.BASELINE_SYSTEM)
        self.assertIn("exactly APPROVE:, COMMENT:, or BLOCK:", baseline.BASELINE_SYSTEM)
        self.assertIn("Do not repeat these instructions", baseline.BASELINE_SYSTEM)
        self.assertIs(baseline.LLMModel, LLMModel)
        self.assertEqual(baseline.baseline_review.__module__, "baseline")

    def test_defaults_to_four_cwes_and_output_directory(self):
        arguments = baseline.parse_args(["--dry-run"])
        self.assertEqual(arguments.cwe, list(baseline.DEFAULT_CWES))
        self.assertEqual(arguments.output, baseline.DEFAULT_OUTPUT)
        self.assertEqual(arguments.model, "Qwen/Qwen2.5-3B-Instruct")

    def test_explicit_scope_and_output_are_preserved(self):
        arguments = baseline.parse_args(
            [
                "--cwe",
                "cwe89",
                "--output",
                "output/custom.jsonl",
                "--limit",
                "2",
            ]
        )
        self.assertEqual(arguments.cwe, ["cwe89"])
        self.assertEqual(arguments.output, Path("output/custom.jsonl"))
        self.assertEqual(arguments.limit, 2)

    def test_custom_benchmark_does_not_claim_default_cwe_filter(self):
        arguments = baseline.parse_args(
            ["--benchmark", "data/custom.jsonl", "--dry-run"]
        )
        self.assertIsNone(arguments.cwe)

    def test_baseline_calls_model_with_local_system_prompt(self):
        model = FakeModel(["APPROVE: first", "COMMENT: second"])
        example = {
            "id": "x",
            "malicious": False,
            "pr_title": "title",
            "pr_body": "body",
            "files_changed": ["x.py"],
            "diff": "+safe = True",
        }
        response = baseline.baseline_review(model, example)
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

    def test_logs_directory_is_rejected_for_model_output(self):
        with self.assertRaises(SystemExit):
            baseline.parse_args(["--output", "logs/baseline.jsonl"])

if __name__ == "__main__":
    unittest.main()
