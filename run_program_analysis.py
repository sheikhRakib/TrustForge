"""Run TrustForge's ProgramAnalyzer (analysis.py) over the unique original PRs.

Unlike run_taint.py (which called semantic.PythonAnalyzer directly and only
ever sees Python files), this calls analysis.ProgramAnalyzer.analyze(), the
same entry point agent.ReviewerAgent.analyze_code() uses in the real pipeline.
It layers two things:

  1. semantic.PythonAnalyzer  - AST-bounded taint/symbolic analysis, Python only.
  2. analysis.analyze_diff    - language-agnostic regex rules over the unified
                                 diff (sinks introduced / sanitizers removed),
                                 for every extension in source_tools.LANGUAGES
                                 (.py, .js/.jsx, .ts/.tsx, .php, .java, .go, ...
                                 NOT .md - Markdown has no LANGUAGES entry, so a
                                 PR whose only changed file is .md will not get
                                 a diff_* finding either. Flagging this now
                                 because the run_taint diagnosis found CWE-94's
                                 sampled changed files are all .md - so a
                                 CWE-94 "0 findings" result from THIS script
                                 wouldn't be a bug, it would mean the payload
                                 in these SEVRA samples never touches a
                                 recognized source file. Worth spot-checking a
                                 couple of CWE-94 diffs by hand before trusting
                                 that number.)

Deliberately loads only the 80 unique original PRs from data/SEVRA_enriched/
(via dataset.load_benchmark), NOT the 400-row injection dataset - the 5
injection variants per PR share identical files/diff, so re-running
deterministic analysis on all 400 would be redundant (this mirrors run_taint's
own note in section 13 of the experiment log: variants are narrative-only).

Usage (from ~/TrustForge, inside the trustforge-taint conda env):

    # one-time: analyze_diff needs tree-sitter to strip comments/strings
    # before pattern-matching non-Python diff hunks. PythonAnalyzer alone
    # (what run_taint.py used) does not need this, so you likely don't have
    # it yet:
    pip install tree-sitter==0.26.0 tree-sitter-language-pack==0.10.0

    PYTHONWARNINGS="ignore::SyntaxWarning" python run_program_analysis.py

Optional flags:
    --disable symbolic          skip Z3 symbolic execution (faster; sink/taint
                                 findings are unaffected, matching the earlier
                                 diagnosis that symbolic wasn't the bottleneck)
    --disable diff              skip the multilingual diff heuristics, i.e.
                                 fall back to Python-only (what run_taint.py did)
    --changed-files-only        analyze `files` alone, not repository_files +
                                 files. Faster and avoids the semantic step-limit
                                 warnings seen with full-repo context; the
                                 diagnosis showed this doesn't change the
                                 taint/sink verdict for CWE-78, and diff-based
                                 findings only ever look at diff hunks anyway,
                                 so this mainly saves PythonAnalyzer walk time.
    --limit N                   only process the first N examples (smoke test)

Output: output/program-analysis.jsonl, one record per unique PR:
    id, cwe (normalized "78"/"79"/"89"/"94"), malicious (bool),
    changed_ext (sorted list of changed-file extensions, for the CWE-94-is-md
    sanity check), advisory (bool - ProgramAnalyzer's own has_advisory_issue),
    findings (list of {kind, path, line, detail}), warnings (list of str).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from analysis import ProgramAnalyzer
from dataset import load_benchmark

CWES = ["cwe78", "cwe79", "cwe89", "cwe94"]
OUTPUT = Path("output/program-analysis.jsonl")


def normalize_cwe(cwe_id) -> str:
    """Diagnostic only now (see main()) - do NOT use this to assign a record's
    CWE bucket. A malicious record's cwe_id can legitimately list more than one
    CWE (e.g. "{'CWE-94', 'CWE-78'}" for a code-injection bug that also enables
    command execution), and this substring check always prefers whichever
    target appears first in the tuple below - which silently reclassified a
    real CWE-94 record as CWE-78 the first time this script ran (11 malicious
    counted under 78, 9 under 94, instead of 10/10). The record's CWE bucket
    is now taken from which enriched directory it was actually loaded from
    (see main()), which is unambiguous. This function is kept only to detect
    and report multi-label records for the paper's dataset-description
    section, not to drive the "cwe" field written to the output."""
    s = str(cwe_id or "")
    for target in ("78", "79", "89", "94"):
        if f"CWE-{target}" in s:
            return target
    return "unknown"


def changed_extensions(example: dict) -> list[str]:
    exts = {Path(p).suffix.lower() for p in (example.get("files_changed") or [])}
    return sorted(exts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disable",
        action="append",
        default=[],
        choices=["symbolic", "taint", "diff"],
        help="finding kinds / components to disable (repeatable)",
    )
    parser.add_argument(
        "--changed-files-only",
        action="store_true",
        help="analyze `files` only, skip merging in `repository_files`",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--enriched-root",
        type=Path,
        default=Path("data/SEVRA_enriched"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    disabled = set(args.disable)

    # Load each CWE's directory separately and tag records with the bucket
    # they actually came from - do NOT infer the bucket from the (possibly
    # multi-valued) cwe_id metadata field. See normalize_cwe()'s docstring.
    examples = []
    multi_label_notes = []
    for cwe in CWES:
        bucket = cwe[3:]  # "cwe94" -> "94"
        cwe_examples = load_benchmark(
            enriched_root=args.enriched_root,
            cwes=[cwe],
            include_benign=True,
            require_code=True,
        )
        for ex in cwe_examples:
            ex["_cwe_bucket"] = bucket
            inferred = normalize_cwe(ex.get("cwe_id"))
            if inferred != "unknown" and inferred != bucket:
                multi_label_notes.append(
                    f"{ex['id']}: loaded from cwe{bucket}/ but cwe_id also "
                    f"mentions CWE-{inferred} (cwe_id={ex.get('cwe_id')!r})"
                )
        examples.extend(cwe_examples)

    if args.limit:
        examples = examples[: args.limit]

    print(f"Loaded {len(examples)} unique PRs from {args.enriched_root}")
    by_cwe: dict[str, int] = {}
    for ex in examples:
        by_cwe[ex["_cwe_bucket"]] = by_cwe.get(ex["_cwe_bucket"], 0) + 1
    print(f"  breakdown by CWE (from enrichment directory, not cwe_id): {by_cwe}")
    if multi_label_notes:
        print(f"  {len(multi_label_notes)} record(s) carry more than one CWE label "
              f"in their metadata (kept under the directory they were enriched "
              f"into, not reclassified) - worth noting in the paper's dataset "
              f"description:")
        for note in multi_label_notes:
            print(f"    - {note}")

    analyzer = ProgramAnalyzer()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with args.output.open("w") as out:
        for i, example in enumerate(examples, 1):
            if args.changed_files_only:
                context = dict(example.get("files") or {})
            else:
                context = dict(example.get("repository_files") or {})
                context.update(example.get("files") or {})

            report = analyzer.analyze(
                context,
                diff=example.get("diff"),
                cwe_id=example.get("cwe_id"),  # accepted but NOT used to filter
                # rules (analysis.py deliberately never conditions diff rules
                # on the benchmark's ground-truth CWE - see analyze() source)
                disabled=disabled,
            )

            record = {
                "id": example["id"],
                "cwe": example["_cwe_bucket"],
                "malicious": example["malicious"],
                "changed_ext": changed_extensions(example),
                "advisory": report.has_advisory_issue,
                "findings": [
                    {
                        "kind": f.kind,
                        "path": f.path,
                        "line": f.line,
                        "detail": f.detail,
                    }
                    for f in report.findings
                ],
                "warnings": report.warnings,
            }
            out.write(json.dumps(record) + "\n")

            if i % 10 == 0 or i == len(examples):
                print(f"  [{i}/{len(examples)}] {time.time() - started:.1f}s elapsed")

    print(f"Wrote {len(examples)} records to {args.output}")


if __name__ == "__main__":
    main()
