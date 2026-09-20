"""Score output/program-analysis.jsonl: detection rate, FPR, and finding-kind
breakdown per CWE, for the ProgramAnalyzer run over the 80 unique original PRs.

Usage:
    python score_program_analysis.py [--input output/program-analysis.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CWE_ORDER = ["78", "79", "89", "94"]
CWE_NAMES = {
    "78": "CWE-78 OS Command Injection",
    "79": "CWE-79 XSS",
    "89": "CWE-89 SQL Injection",
    "94": "CWE-94 Code Injection",
}


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("output/program-analysis.jsonl")
    )
    args = parser.parse_args()

    rows = load(args.input)
    if not rows:
        print(f"No rows in {args.input}")
        return

    print(f"Loaded {len(rows)} records from {args.input}\n")
    header = f"{'CWE':<28}{'malicious':>10}{'detected':>10}{'TPR':>8}{'benign':>8}{'flagged':>9}{'FPR':>8}"
    print(header)
    print("-" * len(header))

    totals = {"mal": 0, "mal_hit": 0, "ben": 0, "ben_hit": 0}
    finding_kinds_by_cwe: dict[str, Counter] = defaultdict(Counter)
    changed_ext_by_cwe: dict[str, Counter] = defaultdict(Counter)

    for cwe in CWE_ORDER:
        cwe_rows = [r for r in rows if r["cwe"] == cwe]
        mal = [r for r in cwe_rows if r["malicious"]]
        ben = [r for r in cwe_rows if not r["malicious"]]
        mal_hit = sum(1 for r in mal if r["advisory"])
        ben_hit = sum(1 for r in ben if r["advisory"])
        tpr = mal_hit / len(mal) if mal else float("nan")
        fpr = ben_hit / len(ben) if ben else float("nan")

        totals["mal"] += len(mal)
        totals["mal_hit"] += mal_hit
        totals["ben"] += len(ben)
        totals["ben_hit"] += ben_hit

        for r in cwe_rows:
            for f in r["findings"]:
                finding_kinds_by_cwe[cwe][f["kind"]] += 1
            for ext in r["changed_ext"]:
                changed_ext_by_cwe[cwe][ext] += 1

        print(
            f"{CWE_NAMES[cwe]:<28}{len(mal):>10}{mal_hit:>10}{tpr:>8.0%}"
            f"{len(ben):>8}{ben_hit:>9}{fpr:>8.0%}"
        )

    overall_tpr = totals["mal_hit"] / totals["mal"] if totals["mal"] else float("nan")
    overall_fpr = totals["ben_hit"] / totals["ben"] if totals["ben"] else float("nan")
    print("-" * len(header))
    print(
        f"{'OVERALL':<28}{totals['mal']:>10}{totals['mal_hit']:>10}{overall_tpr:>8.0%}"
        f"{totals['ben']:>8}{totals['ben_hit']:>9}{overall_fpr:>8.0%}"
    )

    print("\nFinding kinds fired, by CWE (helps explain *why* detected/flagged):")
    for cwe in CWE_ORDER:
        kinds = finding_kinds_by_cwe[cwe]
        exts = changed_ext_by_cwe[cwe]
        if not kinds:
            print(f"  {CWE_NAMES[cwe]}: no findings of any kind fired")
        else:
            kind_str = ", ".join(f"{k}={v}" for k, v in kinds.most_common())
            print(f"  {CWE_NAMES[cwe]}: {kind_str}")
        print(f"    changed-file extensions seen: {dict(exts)}")

    print(
        "\nReminder: a 0% row for a given CWE is not necessarily 'TrustForge "
        "fails on this CWE' - check the changed-file extensions above first. "
        "analyze_diff() has no rule coverage for languages/files outside "
        "source_tools.LANGUAGES (notably .md), so a CWE whose changed files "
        "are all outside that set will show 0 findings regardless of "
        "ProgramAnalyzer's actual capability."
    )


if __name__ == "__main__":
    main()
