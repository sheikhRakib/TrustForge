import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from report import load_results, main as report_main, write_attack_type_figure


class ReportCompletionTests(unittest.TestCase):
    def test_observed_modes_must_cover_the_same_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.jsonl"
            rows = [
                {"id": "one", "malicious": True, "mode": "baseline"},
                {"id": "two", "malicious": True, "mode": "baseline"},
                {"id": "one", "malicious": True, "mode": "hybrid"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "different examples"):
                load_results(path)
            self.assertFalse(load_results(path, allow_partial=True)[1])
            with path.open("a") as handle:
                handle.write(json.dumps(dict(rows[1], mode="hybrid")) + "\n")
            self.assertTrue(load_results(path)[1])

    def test_unmatched_examples_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.jsonl"
            rows = [
                {"id": "one", "malicious": True, "mode": "baseline"},
                {"id": "two", "malicious": True, "mode": "hybrid"},
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                load_results(path)
            path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_results(path, allow_partial=True)

    def test_attack_type_figure_compares_baseline_and_hybrid_only(self):
        rows = []
        for mode, verdicts in {
            "baseline": ("APPROVE", "APPROVE"),
            "multi_agent": ("COMMENT", "APPROVE"),
            "hybrid": ("BLOCK", "COMMENT"),
        }.items():
            for index, verdict in enumerate(verdicts):
                rows.append({
                    "id": f"pr-{index}", "malicious": True, "mode": mode,
                    "framing": "appeal_to_authority", "variant": "original",
                    "source": "sevra", "verdict": verdict,
                })
            rows.append({
                "id": "benign", "malicious": False, "mode": mode,
                "framing": None, "variant": "original", "source": "sevra",
                "verdict": "APPROVE",
            })
            rows.append({
                "id": "synthetic", "malicious": True, "mode": mode,
                "framing": "appeal_to_authority", "variant": "base64",
                "source": "synthetic", "verdict": "APPROVE",
            })
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            self.assertTrue(write_attack_type_figure(rows, output, complete=True))
            with (output / "attack-type-asr.csv").open() as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 1)
            self.assertEqual(metrics[0]["baseline_asr"], "1.0")
            self.assertNotIn("multi_agent_asr", metrics[0])
            self.assertEqual(metrics[0]["hybrid_asr"], "0.0")
            self.assertEqual(metrics[0]["hybrid_malicious_n"], "2")
            self.assertTrue((output / "attack-type-asr.png").exists())
            self.assertFalse((output / "attack-type-asr.pdf").exists())
            self.assertFalse((output / "attack-type-asr.svg").exists())

    def test_report_does_not_generate_variant_asr_figure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = root / "results.jsonl"
            rows = [
                {"id": "pr", "malicious": True, "mode": mode,
                 "framing": "fake_bug_fix", "variant": "original",
                 "source": "sevra", "verdict": verdict}
                for mode, verdict in (("baseline", "APPROVE"), ("hybrid", "BLOCK"))
            ]
            results.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "report"
            with patch("sys.argv", ["report.py", str(results), "--output-dir", str(output)]):
                report_main()
            self.assertTrue((output / "attack-type-asr.png").exists())
            self.assertTrue((output / "metrics.csv").exists())
            self.assertEqual(list(output.glob("asr_sevra.*")), [])
