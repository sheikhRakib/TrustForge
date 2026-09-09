import difflib
import unittest

from analysis import analyze_diff
from source_tools import strip_source


def patch(path, before, after):
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/" + path,
            tofile="b/" + path,
        )
    )


class LiteralHeuristicTests(unittest.TestCase):
    def test_inert_literal_does_not_raise_sink_or_removed_guard(self):
        for path, before, after in [
            ("a.py", 'x = "csrf_protect"\n', 'x = "subprocess.run"\n'),
            ("a.js", 'let x = "csrf_protect";\n', 'let x = "subprocess.run";\n'),
            ("a.go", "x := `csrf_protect`\n", "x := `subprocess.run`\n"),
            ("a.c", 'char *x = "csrf_protect";\n', 'char *x = "subprocess.run";\n'),
        ]:
            with self.subTest(path=path):
                self.assertEqual(analyze_diff(patch(path, before, after)), [])

    def test_real_call_still_detected(self):
        findings = analyze_diff(patch("a.py", "", "subprocess.run(input())\n"))
        self.assertTrue(any(f.kind == "diff_sink" for f in findings))

    def test_sql_construction_and_bound_parameters(self):
        safe = 'cursor.execute("SELECT * FROM users WHERE id=?", (value,))\n'
        unsafe = 'cursor.execute("SELECT * FROM users WHERE id=" + value)\n'
        self.assertEqual(analyze_diff(patch("a.py", "", safe)), [])
        self.assertTrue(analyze_diff(patch("a.py", "", unsafe)))

    def test_interpolation_keeps_executable_calls(self):
        for path, source in [
            ("a.py", 'x = f"subprocess.run {eval(input())}"\n'),
            ("a.js", "let x = `subprocess.run ${eval(input)}`;\n"),
            ("a.rb", 'x = "subprocess.run #{eval(input)}"\n'),
        ]:
            with self.subTest(path=path):
                clean, _ = strip_source(source, path, mask_literal_text=True)
                self.assertNotIn("subprocess.run", clean)
                self.assertIn("eval(input", clean)
                self.assertEqual(source.count("\n"), clean.count("\n"))
                self.assertIn("subprocess.run", strip_source(source, path)[0])
