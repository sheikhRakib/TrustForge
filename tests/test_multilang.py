"""Multilingual sink/taint heuristic regressions."""

import unittest

from analysis import ProgramAnalyzer
from multilang import analyze_file


class MultilangTests(unittest.TestCase):
    def test_php_source_and_sink_colocated(self):
        findings, _ = analyze_file(
            "a.php",
            "<?php\nfunction f() {\n  $x = $_GET['q'];\n  eval($x);\n}\n",
        )
        self.assertTrue(any(f["kind"] == "sink" and "eval" in f["detail"] for f in findings))
        self.assertTrue(any(f["kind"] == "taint" for f in findings))

    def test_c_high_risk_sink_without_source_is_advisory(self):
        findings, _ = analyze_file(
            "a.c",
            "void f(char *p) {\n  strcpy(buf, p);\n}\n",
        )
        self.assertTrue(any(f["kind"] == "sink" for f in findings))
        self.assertTrue(
            any(
                f["kind"] == "taint" and "high-risk" in f["detail"]
                for f in findings
            )
        )

    def test_go_query_with_form_value(self):
        findings, _ = analyze_file(
            "a.go",
            "func f(r *http.Request, db *sql.DB) {\n"
            "  q := r.FormValue(\"id\")\n"
            "  db.Query(q)\n"
            "}\n",
        )
        self.assertTrue(any("Query" in f["detail"] for f in findings if f["kind"] == "sink"))
        self.assertTrue(any(f["kind"] == "taint" for f in findings))

    def test_java_exec_detected(self):
        findings, _ = analyze_file(
            "A.java",
            "class A {\n"
            "  void f(String q) {\n"
            "    Runtime.getRuntime().exec(q);\n"
            "  }\n"
            "}\n",
        )
        self.assertTrue(any("exec" in f["detail"] for f in findings))

    def test_program_analyzer_merges_python_and_multilang(self):
        report = ProgramAnalyzer().analyze(
            {
                "a.py": "eval(input())\n",
                "b.js": "eval(req.body);\n",
            }
        )
        paths = {f.path for f in report.findings}
        self.assertIn("a.py", paths)
        self.assertIn("b.js", paths)


if __name__ == "__main__":
    unittest.main()
