"""Regression tests for the modeled non-Python flow subset."""

import unittest
from unittest.mock import patch

from analysis import ProgramAnalyzer
from agent import ReviewerAgent


def kinds(files, *, disabled=()):
    report = ProgramAnalyzer().analyze(files, disabled=disabled)
    return {finding.kind for finding in report.findings}, report


class FlowAnalysisTests(unittest.TestCase):
    def test_javascript_tracks_assignment_and_kill(self):
        flowing, _ = kinds({"a.js": "const x = req.body.cmd; eval(x);"})
        safe, _ = kinds({"a.js": "let x = req.body.cmd; x = 'safe'; eval(x);"})
        self.assertIn("taint", flowing)
        self.assertIn("symbolic", flowing)
        self.assertNotIn("taint", safe)
        self.assertIn("sink", safe)

    def test_bound_sql_value_does_not_taint_query_string(self):
        found, _ = kinds({"a.js": "db.query('SELECT * FROM t WHERE id = ?', [req.body.id]);"})
        self.assertNotIn("taint", found)
        self.assertIn("sink", found)

    def test_xss_output_sinks_are_traced(self):
        js, _ = kinds({"a.js": "const x = req.body.msg; element.innerHTML = x;"})
        php, _ = kinds({"a.php": "<?php $x = $_GET['msg']; echo $x;"})
        self.assertIn("taint", js)
        self.assertIn("sink", js)
        self.assertIn("taint", php)
        self.assertIn("sink", php)

    def test_symbolic_branch_witness_and_infeasible_path(self):
        _, report = kinds({
            "a.ts": "const x = req.body.cmd; if (x === 'bad') { eval(x); }"
        })
        witnesses = [finding.witness for finding in report.findings
                     if finding.kind == "symbolic"]
        self.assertTrue(any("bad" in witness.values() for witness in witnesses))
        impossible, _ = kinds({
            "a.ts": "const x = req.body.cmd; if (x === 'bad') {"
                    " if (x === 'good') { eval(x); } }"
        })
        self.assertNotIn("taint", impossible)
        repeated, _ = kinds({
            "a.ts": "if (req.body.cmd === 'bad') {"
                    " if (req.body.cmd === 'good') { eval(req.body.cmd); } }"
        })
        self.assertNotIn("taint", repeated)

    def test_symbolic_ablation_keeps_taint(self):
        with patch("semantic.z3.Solver", side_effect=AssertionError("solver invoked")):
            found, _ = kinds({"a.js": "eval(req.body.cmd);"}, disabled=("symbolic",))
        self.assertIn("taint", found)
        self.assertNotIn("symbolic", found)
        no_taint, _ = kinds({"a.js": "eval(req.body.cmd);"}, disabled=("taint",))
        self.assertNotIn("taint", no_taint)
        self.assertNotIn("symbolic", no_taint)

    def test_unsupported_condition_does_not_claim_symbolic_witness(self):
        found, _ = kinds({
            "a.js": "const x = req.body.cmd; if (allow(x)) { eval(x); }"
        })
        self.assertIn("taint", found)
        self.assertNotIn("symbolic", found)

    def test_javascript_import_alias_cross_file_flow(self):
        files = {
            "src/main.js": "import {go as review} from './helper.js'; review(req.body.cmd);",
            "src/helper.js": "export function go(x) { eval(x); }",
        }
        found, report = kinds(files)
        self.assertIn("cross_file", found)
        self.assertTrue(any(f.kind == "taint" and f.path == "src/helper.js"
                            for f in report.findings))
        disabled, _ = kinds(files, disabled=("cross_file",))
        self.assertNotIn("cross_file", disabled)
        self.assertNotIn("taint", disabled)

    def test_php_include_cross_file_flow(self):
        files = {
            "main.php": "<?php require_once 'helper.php'; go($_GET['q']);",
            "helper.php": "<?php function go($x) { eval($x); }",
        }
        found, _ = kinds(files)
        self.assertIn("taint", found)
        self.assertIn("cross_file", found)
        self.assertIn("symbolic", found)

    def test_unavailable_import_is_explicit(self):
        _, report = kinds({
            "main.js": "import {go} from './missing.js'; go(req.body.cmd);"
        })
        self.assertTrue(any("import target unavailable" in note
                            for note in report.warnings))

    def test_unmodeled_loop_is_reported_without_fake_trace(self):
        found, report = kinds({"a.js": "for (const x of req.body.items) { eval(x); }"})
        self.assertIn("sink", found)
        self.assertNotIn("taint", found)
        self.assertTrue(any("flow not modeled" in note for note in report.warnings))

    def test_missing_repository_context_is_reported(self):
        report = ReviewerAgent(None).analyze_code({
            "files": {"a.js": "eval(req.body.cmd);"}, "diff": ""
        })
        self.assertTrue(any("Unchanged repository context unavailable" in note
                            for note in report.warnings))

    def test_large_generated_source_is_skipped_explicitly(self):
        found, report = kinds({"bundle.min.js": " " * 1_000_001 + "eval(req.body);"})
        self.assertNotIn("taint", found)
        self.assertTrue(any("source exceeds" in note for note in report.warnings))


if __name__ == "__main__":
    unittest.main()
