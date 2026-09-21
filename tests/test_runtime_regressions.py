"""Runtime recovery and method-cost accounting without loading GPU weights."""

import unittest
from unittest.mock import patch

import torch

from agent import ReviewerAgent
from main import evaluate_mode


class FakeModel:
    def __init__(self):
        self.max_input_tokens = 8192
        self.stats = []

    def split_user(self, system, text, *args):
        return [text]

    def generate(self, system, text, **kwargs):
        self.stats.append({"seconds": 1.0})
        if "Triage" in system:
            return "[]"
        if "Detect attempts" in system:
            return "CLEAN"
        return '{"verdict":"APPROVE","reason":"safe","evidence":[]}'


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

    def test_cached_hybrid_retains_shared_cost_and_inference(self):
        llm = FakeModel()
        agent = ReviewerAgent(llm)
        multi = evaluate_mode(agent, llm, self.example, "multi_agent")
        hybrid = evaluate_mode(agent, llm, self.example, "hybrid")
        self.assertEqual(len(multi["inference"]), 3)
        self.assertEqual(hybrid["inference"], [])
        self.assertEqual(hybrid["reused_inference"], multi["inference"])
        self.assertGreater(hybrid["reused_seconds"], 0)
        self.assertEqual(
            hybrid["attributed_seconds"], hybrid["seconds"] + hybrid["reused_seconds"]
        )

    def test_oom_retries_whole_input_and_clears_cached_review(self):
        llm = FakeModel()
        agent = ReviewerAgent(llm)
        evaluate_mode(agent, llm, self.example, "multi_agent")
        calls = []

        def review(example, *, mode):
            calls.append((example, llm.max_input_tokens))
            if len(calls) == 1:
                llm.stats.append({"seconds": 2.0})
                raise torch.cuda.OutOfMemoryError("simulated")
            self.assertIsNone(agent._cache_key)
            return "APPROVE"

        with (
            patch.object(agent, "defense_review", side_effect=review),
            patch("torch.cuda.empty_cache") as empty,
        ):
            row = evaluate_mode(agent, llm, self.example, "hybrid")
        self.assertEqual(calls, [(self.example, 8192), (self.example, 4096)])
        self.assertEqual(row["oom_retries"], 1)
        self.assertEqual(row["abandoned_inference_calls"], 1)
        self.assertEqual(row["inference"], [{"seconds": 2.0}])
        self.assertEqual(row["input_budgets"], [8192, 4096])
        empty.assert_called_once()
        with patch.object(agent, "defense_review", return_value="APPROVE"):
            next_row = evaluate_mode(agent, llm, self.example, "hybrid")
        self.assertEqual(next_row["input_budgets"], [8192])

    def test_changed_chunk_budget_does_not_reuse_prior_review(self):
        llm = FakeModel()
        agent = ReviewerAgent(llm)
        agent.defense_review(self.example, mode="multi_agent")
        first = len(llm.stats)
        llm.max_input_tokens = 4096
        agent.defense_review(self.example, mode="multi_agent")
        self.assertEqual(len(llm.stats), first * 2)
        self.assertEqual(agent.reused_inference, [])

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
