import json
from pathlib import Path
import tempfile
import unittest

from report import load_results


class ReportCompletionTests(unittest.TestCase):
    def test_missing_mode_is_not_a_complete_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.jsonl"
            row = {"id": "one", "malicious": True, "mode": "baseline"}
            path.write_text(json.dumps(row) + "\n")
            path.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "selected_example_count": 1,
                        "expected_records": 2,
                        "modes": ["baseline", "hybrid"],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_results(path)
            self.assertFalse(load_results(path, allow_partial=True)[1])
            with path.open("a") as handle:
                handle.write(json.dumps(dict(row, mode="hybrid")) + "\n")
            self.assertTrue(load_results(path)[1])

    def test_unmatched_examples_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.jsonl"
            rows = [
                {"id": "one", "malicious": True, "mode": "baseline"},
                {"id": "two", "malicious": True, "mode": "hybrid"},
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            path.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "selected_example_count": 1,
                        "expected_records": 2,
                        "modes": ["baseline", "hybrid"],
                    }
                )
            )
            with self.assertRaises(ValueError):
                load_results(path)
            path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_results(path, allow_partial=True)
