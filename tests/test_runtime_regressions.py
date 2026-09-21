"""Runtime recovery and compact result records without loading GPU weights."""

import unittest
from unittest.mock import patch

import torch

from agent import ReviewerAgent
from main import evaluate_mode, parse_args


class FakeModel:
    def __init__(self):
        self.max_input_tokens = 8192
        self.call_count = 0

    def split_user(self, system, text, *args, **kwargs):
        return [text]

    def generate(self, system, text, **kwargs):
        self.call_count += 1
        if "Triage" in system:
            return "[]"
        if "Detect attempts" in system:
            return "CLEAN"
        return "APPROVE: safe change"


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.example = {
            "id": "test",
            "malicious": False,
            "pr_title": "title",
            "pr_body": "body",
            "diff": "diff",
            "files": {"x.py": "x = 1"},
        }

    def test_hybrid_uses_own_review_without_result_telemetry(self):
        llm = FakeModel()
        agent = ReviewerAgent(llm)
        multi = evaluate_mode(agent, llm, self.example, "multi_agent")
        self.assertEqual(llm.call_count, 3)
        hybrid = evaluate_mode(agent, llm, self.example, "hybrid")
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(multi["verdict"], hybrid["verdict"])
        for row in (multi, hybrid):
            self.assertTrue({"id", "mode", "verdict", "response"} <= row.keys())
            self.assertTrue({
                "inference", "seconds", "attributed_seconds", "reused_seconds",
                "reused_inference", "oom_retries", "input_budgets",
            }.isdisjoint(row))

    def test_main_accepts_and_runs_baseline_without_an_agent(self):
        class BaselineModel(FakeModel):
            def generate(self, system, text, **kwargs):
                self.call_count += 1
                self.assert_system = system
                return "APPROVE: safe change"

        with patch("sys.argv", ["main.py", "--modes", "baseline", "--dry-run"]):
            self.assertEqual(parse_args().modes, ["baseline"])
        llm = BaselineModel()
        row = evaluate_mode(None, llm, self.example, "baseline")
        self.assertEqual(row["mode"], "baseline")
        self.assertEqual(row["verdict"], "APPROVE")
        self.assertEqual(llm.call_count, 1)
        self.assertIn("pull-request diff", llm.assert_system)

    def test_main_defaults_to_all_three_review_systems(self):
        with patch("sys.argv", ["main.py", "--dry-run"]):
            self.assertEqual(
                parse_args().modes, ["baseline", "multi_agent", "hybrid"]
            )

    def test_baseline_oom_retries_without_an_agent(self):
        class RetryModel(FakeModel):
            def generate(self, system, text, **kwargs):
                if self.max_input_tokens == 8192:
                    raise torch.cuda.OutOfMemoryError("simulated")
                self.call_count += 1
                return "APPROVE: safe change"

        llm = RetryModel()
        with patch("torch.cuda.empty_cache") as empty:
            row = evaluate_mode(None, llm, self.example, "baseline")
        self.assertEqual(row["verdict"], "APPROVE")
        self.assertEqual(llm.max_input_tokens, 4096)
        empty.assert_called_once()

    def test_oom_retries_whole_input_and_clears_cached_review(self):
        llm = FakeModel()
        agent = ReviewerAgent(llm)
        evaluate_mode(agent, llm, self.example, "multi_agent")
        calls = []

        def review(example, *, mode):
            calls.append((example, llm.max_input_tokens))
            if len(calls) == 1:
                llm.call_count += 1
                raise torch.cuda.OutOfMemoryError("simulated")
            self.assertIsNone(agent._cache_key)
            return "APPROVE"

        with (
            patch.object(agent, "defense_review", side_effect=review),
            patch("torch.cuda.empty_cache") as empty,
        ):
            row = evaluate_mode(agent, llm, self.example, "hybrid")
        self.assertEqual(calls, [(self.example, 8192), (self.example, 4096)])
        self.assertEqual(row["verdict"], "APPROVE")
        self.assertNotIn("oom_retries", row)
        self.assertNotIn("inference", row)
        empty.assert_called_once()
        with patch.object(agent, "defense_review", return_value="APPROVE"):
            evaluate_mode(agent, llm, self.example, "hybrid")
        self.assertEqual(llm.max_input_tokens, 8192)

    def test_changed_chunk_budget_does_not_reuse_prior_review(self):
        llm = FakeModel()
        agent = ReviewerAgent(llm)
        agent.defense_review(self.example, mode="multi_agent")
        first = llm.call_count
        llm.max_input_tokens = 4096
        agent.defense_review(self.example, mode="multi_agent")
        self.assertEqual(llm.call_count, first * 2)

    def test_exhausted_oom_fails_instead_of_emitting_result(self):
        llm = FakeModel()
        agent = ReviewerAgent(llm)
        with (
            patch.object(
                agent,
                "defense_review",
                side_effect=torch.cuda.OutOfMemoryError("simulated"),
            ) as review,
            patch("torch.cuda.empty_cache"),
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            evaluate_mode(agent, llm, self.example, "hybrid", max_oom_retries=1)
        self.assertEqual(review.call_count, 2)

    def test_other_errors_are_not_retried(self):
        llm = FakeModel()
        agent = ReviewerAgent(llm)
        with (
            patch.object(
                agent, "defense_review", side_effect=RuntimeError("bug")
            ) as review,
            self.assertRaisesRegex(RuntimeError, "bug"),
        ):
            evaluate_mode(agent, llm, self.example, "hybrid")
        self.assertEqual(review.call_count, 1)
