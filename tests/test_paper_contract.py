"""Paper-facing behavior and ablation contracts."""

import unittest
from unittest.mock import patch

from agent import ReviewerAgent
from analysis import AnalysisReport, Finding, ProgramAnalyzer
from augment import semantic_fixtures, variants
from main import DEFAULT_OUTPUT, SLURM_LOG_DIR, is_slurm_log_path
from report import DEFAULT_REPORT_DIR
from semantic import PythonAnalyzer


class AggregatorPolicyTests(unittest.TestCase):
    def setUp(self):
        self.agent = ReviewerAgent(None)
        self.approved = {
            "scan": [],
            "injection": {"detected": False, "valid": True, "note": "CLEAN"},
            "auditor": {"reviews": [{"verdict": "APPROVE", "grounded": False}]},
        }

    def test_analysis_is_advisory_without_grounded_auditor_defect(self):
        report = AnalysisReport(
            findings=[Finding("taint", "x.py", 1, "modeled flow")]
        )
        result = self.agent.aggregate(self.approved, report)
        self.assertEqual(result["verdict"], "COMMENT")

    def test_injection_detection_alone_cannot_force_rejection(self):
        self.approved["injection"]["detected"] = True
        result = self.agent.aggregate(self.approved, use_analysis=False)
        self.assertEqual(result["verdict"], "APPROVE")


class AblationContractTests(unittest.TestCase):
    SOURCE = "x = int(input())\nif x == 7:\n    eval(x)\n"

    def test_symbolic_ablation_keeps_taint_but_never_invokes_solver(self):
        with patch("semantic.z3.Solver", side_effect=AssertionError("solver invoked")):
            findings, _ = PythonAnalyzer(
                {"x.py": self.SOURCE}, disabled={"symbolic"}
            ).run()
        self.assertTrue(any(f["kind"] == "taint" for f in findings))
        self.assertFalse(any(f["kind"] == "symbolic" for f in findings))

    def test_taint_ablation_removes_taint_and_dependent_symbolic(self):
        with patch("semantic.z3.Solver", side_effect=AssertionError("solver invoked")):
            findings, _ = PythonAnalyzer(
                {"x.py": self.SOURCE}, disabled={"taint"}
            ).run()
        self.assertFalse(any(f["kind"] in {"taint", "symbolic"} for f in findings))
        self.assertTrue(any(f["kind"] == "sink" for f in findings))

    def test_non_python_semantics_use_multilang_heuristics(self):
        report = ProgramAnalyzer().analyze(
            {"x.js": "const x = req.body; eval(x);\n"}
        )
        self.assertTrue(any(f.kind == "sink" for f in report.findings))
        self.assertTrue(any(f.kind == "taint" for f in report.findings))
        self.assertTrue(
            any("multilingual sink/taint heuristics" in w for w in report.warnings)
        )
        self.assertTrue(
            any("Python-only" in w for w in report.warnings)
        )

    def test_non_python_taint_ablation_keeps_sink_inventory(self):
        report = ProgramAnalyzer().analyze(
            {"x.js": "eval(req.body);\n"}, disabled=("taint",)
        )
        self.assertTrue(any(f.kind == "sink" for f in report.findings))
        self.assertFalse(any(f.kind == "taint" for f in report.findings))


class VariantScopeTests(unittest.TestCase):
    def test_sevra_variants_are_explicitly_narrative_only(self):
        example = {
            "id": "sevra:x",
            "malicious": True,
            "pr_body": "body",
            "files": {"x.py": "eval(input())"},
            "diff": "diff",
        }
        generated = list(variants(example))
        self.assertTrue(
            all(
                item["transformation_scope"] == "sevra_narrative_only"
                and item["code_transformed"] is False
                and item["files"] == example["files"]
                and item["diff"] == example["diff"]
                for item in generated
            )
        )

    def test_semantic_cases_cannot_be_mistaken_for_sevra_transforms(self):
        fixtures = list(semantic_fixtures())
        self.assertTrue(
            all(
                item["source"] == "synthetic"
                and item["transformation_scope"]
                == "synthetic_not_sevra_transform"
                and item["code_transformed"] is True
                for item in fixtures
            )
        )


class OutputLocationTests(unittest.TestCase):
    def test_model_and_report_outputs_default_outside_slurm_logs(self):
        self.assertEqual(DEFAULT_OUTPUT.parts[0], "output")
        self.assertEqual(DEFAULT_REPORT_DIR.parts[0], "output")
        self.assertTrue(is_slurm_log_path(SLURM_LOG_DIR / "x.jsonl"))
        self.assertFalse(is_slurm_log_path(DEFAULT_OUTPUT))

    def test_default_model_and_input_budget_match_qwen25_3b(self):
        from main import DEFAULT_MAX_INPUT_TOKENS, MODEL_NAME

        self.assertEqual(MODEL_NAME, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(DEFAULT_MAX_INPUT_TOKENS, 32_768)


if __name__ == "__main__":
    unittest.main()
