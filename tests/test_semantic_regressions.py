"""Regression cases for supported sinks and explicit semantic coverage."""

import unittest

from semantic import PythonAnalyzer


class SemanticRegressionTests(unittest.TestCase):
    def analyze(self, source):
        return PythonAnalyzer({"example.py": source}).run()

    def test_bound_sql_values_are_not_query_taint(self):
        for method in ("execute", "executemany"):
            with self.subTest(method=method):
                findings, _ = self.analyze(
                    'x = input()\n'
                    f'cursor.{method}("SELECT * FROM users WHERE name = ?", (x,))\n'
                )
                self.assertTrue(any(f["kind"] == "sink" for f in findings))
                self.assertFalse(any(f["kind"] in {"taint", "symbolic"} for f in findings))

    def test_interpolated_sql_remains_tainted_with_bound_values(self):
        findings, _ = self.analyze(
            'x = input()\n'
            'cursor.execute("SELECT * FROM " + x + " WHERE id = ?", (1,))\n'
        )
        self.assertTrue(any(f["kind"] == "taint" for f in findings))
        self.assertTrue(any(f["kind"] == "symbolic" for f in findings))

    def test_keyword_sql_query_is_distinguished_from_parameters(self):
        for query, expected in [('"SELECT * FROM users WHERE name = ?"', False), ('x', True)]:
            with self.subTest(query=query):
                findings, _ = self.analyze(
                    f'x = input()\ncursor.execute(query={query}, params=(x,))\n'
                )
                self.assertEqual(any(f["kind"] == "taint" for f in findings), expected)

    def test_sql_argument_side_effects_are_still_analyzed(self):
        findings, _ = self.analyze(
            'cursor.execute("SELECT ?", (eval(input()),))\n'
        )
        taints = [f for f in findings if f["kind"] == "taint"]
        self.assertTrue(any("eval" in f["detail"] for f in taints))
        self.assertFalse(any("cursor.execute" in f["detail"] for f in taints))

    def test_request_parameter_is_a_scoped_explicit_assumption(self):
        findings, warnings = self.analyze(
            'def endpoint(request):\n'
            '    eval(request.args.get("expression"))\n'
        )
        self.assertTrue(any(f["kind"] == "taint" for f in findings))
        self.assertTrue(any("assumed to be an HTTP request" in w for w in warnings))

    def test_unknown_object_is_not_implicitly_a_request(self):
        findings, warnings = self.analyze(
            'def endpoint(other):\n'
            '    eval(other.args.get("expression"))\n'
        )
        self.assertFalse(any(f["kind"] == "taint" for f in findings))
        self.assertTrue(any("unsupported call" in w for w in warnings))

    def test_request_assignment_clears_assumed_identity(self):
        findings, _ = self.analyze(
            'def endpoint(request):\n'
            '    request = 1\n'
            '    eval(request.args.get("expression"))\n'
        )
        self.assertFalse(any(f["kind"] == "taint" for f in findings))

    def test_imported_flask_request_alias_is_a_source(self):
        findings, _ = self.analyze(
            'from flask import request as incoming\n'
            'eval(incoming.args.get("expression"))\n'
        )
        self.assertTrue(any(f["kind"] == "taint" for f in findings))

    def test_class_methods_report_missing_semantic_coverage(self):
        findings, warnings = self.analyze(
            'class Handler:\n'
            '    def handle(self):\n'
            '        eval(input())\n'
        )
        self.assertTrue(any(f["kind"] == "sink" for f in findings))
        self.assertTrue(any("class body/method semantics unsupported" in w for w in warnings))
        self.assertTrue(any("nested function/method semantics unsupported" in w for w in warnings))

    def test_nested_function_reports_missing_semantic_coverage(self):
        _, warnings = self.analyze(
            'def outer():\n'
            '    def inner():\n'
            '        eval(input())\n'
        )
        self.assertTrue(any("nested function/method semantics unsupported" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
