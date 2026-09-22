"""Concern reassessment and evidence transfer, without GPU inference."""

import unittest

from agent import CONCERN_SYSTEM, ReviewerAgent
from analysis import AnalysisReport, Finding


class ScriptedModel:
    def __init__(self, initial, checked, split_check=False):
        self.initial = initial
        self.checked = checked
        self.split_check = split_check
        self.calls = []

    def split_user(self, system, text, max_new_tokens):
        if system == CONCERN_SYSTEM and self.split_check:
            return [text[:10], text[10:]]
        return [text]

    def generate(self, system, text, max_new_tokens):
        self.calls.append((system, text))
        return self.checked if system == CONCERN_SYSTEM else self.initial


class ConcernTests(unittest.TestCase):
    def review(self, model, source="eval(input())"):
        agent = ReviewerAgent(model)
        diff = "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+" + source + "\n"
        audit = agent.auditor({"app.py": source}, diff=diff)
        return audit["reviews"][0], agent.aggregate({"auditor": audit}, use_analysis=False)

    def test_inline_citation_and_legacy_citation_resolve_identically(self):
        agent = ReviewerAgent(None)
        visible = {"L1": ("app.py", 1, "eval(input())")}
        for text in (
            "BLOCK: input reaches eval | EVIDENCE: L1",
            "BLOCK: input reaches eval\nEVIDENCE: L1",
            "BLOCK: EVIDENCE: L1 input reaches eval",
        ):
            result = agent.inspector(text, visible)
            self.assertEqual(result["verdict"], "BLOCK")
            self.assertTrue(result["grounded"])
            self.assertEqual(result["reason"], "input reaches eval")

    def test_citation_without_explanation_is_not_a_security_finding(self):
        result = ReviewerAgent(None).inspector(
            "BLOCK: EVIDENCE: L1", {"L1": ("app.py", 1, "x = 1")}
        )
        self.assertEqual(result["verdict"], "UNKNOWN")

    def test_unseen_or_ambiguous_citations_cannot_ground_a_block(self):
        agent = ReviewerAgent(None)
        visible = {"L1": ("app.py", 1, "eval(input())")}
        for text in (
            "BLOCK: input reaches eval | EVIDENCE: L99",
            "BLOCK: input reaches eval | EVIDENCE: L1 | EVIDENCE: L2",
        ):
            self.assertFalse(agent.inspector(text, visible)["grounded"])

    def test_reassessment_can_clear_a_false_alarm(self):
        model = ScriptedModel(
            "COMMENT: dependency might be vulnerable | EVIDENCE: L1",
            "APPROVE: version update has no identified vulnerability",
        )
        result, final = self.review(model, 'version = "8.0.1"')
        self.assertEqual(final["verdict"], "APPROVE")
        self.assertEqual(result["initial_review"]["verdict"], "COMMENT")
        self.assertIn('+version = "8.0.1"', model.calls[1][1])
        self.assertIn("dependency might be vulnerable", model.calls[1][1])
        self.assertEqual(len(model.calls), 2)

    def test_reassessment_keeps_a_demonstrated_defect(self):
        model = ScriptedModel(
            "BLOCK: input reaches eval | EVIDENCE: L1",
            "BLOCK: arbitrary user code executes via eval | EVIDENCE: L1",
        )
        result, final = self.review(model)
        self.assertEqual(final["verdict"], "BLOCK")
        self.assertTrue(result["grounded"])

    def test_failed_verification_never_becomes_approval(self):
        for response in ("I cannot decide", "BLOCK: EVIDENCE: L1"):
            with self.subTest(response=response):
                result, final = self.review(ScriptedModel(
                    "BLOCK: input reaches eval | EVIDENCE: L1", response,
                ))
                self.assertEqual(final["verdict"], "BLOCK")
                self.assertEqual(result["verification"]["status"], "invalid")

    def test_context_limit_retains_original_concern_without_another_call(self):
        model = ScriptedModel(
            "COMMENT: input provenance is missing | EVIDENCE: L1",
            "APPROVE: safe", split_check=True,
        )
        result, final = self.review(model)
        self.assertEqual(final["verdict"], "COMMENT")
        self.assertEqual(result["verification"]["status"], "skipped_context_limit")
        self.assertEqual(len(model.calls), 1)

    def test_approval_has_no_extra_call(self):
        model = ScriptedModel("APPROVE: constant assignment", "unused")
        _, final = self.review(model, "x = 1")
        self.assertEqual(final["verdict"], "APPROVE")
        self.assertEqual(len(model.calls), 1)

    def test_hybrid_explanations_accompany_every_chunk_as_untrusted_content(self):
        class ChunkedModel:
            def __init__(self):
                self.calls = []

            def split_user(self, system, text, max_new_tokens):
                return ["patch chunk one", "patch chunk two"]

            def generate(self, system, text, max_new_tokens):
                self.calls.append((system, text))
                return "APPROVE: patch supplies the missing validation"

        model = ChunkedModel()
        agent = ReviewerAgent(model)
        agent.multi_agent_review = lambda example: {
            "scan": [], "injection": {"detected": False},
            "auditor": {"reviews": [{
                "verdict": "COMMENT", "reason": "user input might reach eval",
                "grounded": False, "evidence": [],
            }]},
        }
        agent.analyze_code = lambda example: AnalysisReport([
            Finding("taint", "app.py", 1, "heuristic flow; validation not modeled")
        ])
        response = agent.hybrid_review({
            "pr_title": "Validate input", "pr_body": "Fix validation", "diff": "+validate(x)",
            "files": {"app.py": "validate(x)"},
        })
        self.assertTrue(response.startswith("APPROVE\n"))
        self.assertEqual(len(model.calls), 2)
        for system, text in model.calls:
            self.assertIn("untrusted", text)
            self.assertIn("user input might reach eval", text)
            self.assertIn("heuristic flow; validation not modeled", text)
            self.assertIn("1: validate(x)", text)
            self.assertIn("patch chunk", text)
            self.assertNotIn("validation not modeled", system)


if __name__ == "__main__":
    unittest.main()
