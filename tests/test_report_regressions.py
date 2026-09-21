import json
from pathlib import Path
import tempfile
import unittest

from report import load_results


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
