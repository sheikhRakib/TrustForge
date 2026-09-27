"""Independent sanity checks for a variant benchmark (and optionally its results).

    python check_variants.py output/injection-variants.jsonl
    python check_variants.py output/injection-variants.jsonl \
        --results output/injection-variants-analysis.jsonl

Re-derives every property from the raw JSONL (does not import variant.py), so a
bug in the generator cannot hide itself. Prints PASS/FAIL per check and exits
non-zero on any failure.
"""

import argparse
import base64
import json
import sys
from collections import Counter, defaultdict

VARIANTS = ["neutral_note", "authority_note", "role_hijack", "base64", "hex", "homoglyph", "split_docs"]
BODY = set(VARIANTS) - {"split_docs"}
failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def decoded_payload(row):
    """Recover the injected instruction text from a variant, as plain text."""
    v = row["variant"]
    if v == "split_docs":
        return " ".join(row["files"][p] for p in row.get("added_files", []))
    tail = row["pr_body"].rsplit("\n\n", 1)[-1]
    if v == "neutral_note":
        return tail
    if v == "base64":
        return base64.b64decode(tail.rsplit(" ", 1)[-1]).decode()
    if v == "hex":
        return bytes.fromhex(tail.rsplit(" ", 1)[-1]).decode()
    if v == "homoglyph":  # map Cyrillic lookalikes back to Latin
        return tail.translate(str.maketrans("АЕОРСаеорс", "AEOPCaeopc"))
    return tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--results")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.benchmark, encoding="utf-8") if l.strip()]
    print(f"{len(rows)} records in {a.benchmark}\n")

    # 1. Shape and uniqueness
    keys = [(r["id"], r["malicious"]) for r in rows]
    check("(id, malicious) keys are unique", len(keys) == len(set(keys)),
          f"{len(keys) - len(set(keys))} duplicates")
    check("malicious is always a real boolean",
          all(type(r["malicious"]) is bool for r in rows))
    empty = [r["id"] for r in rows
             if not all((r.get(f) or "").strip() for f in ("pr_title", "pr_body", "diff"))]
    check("no empty title/description/diff", not empty, f"{len(empty)} e.g. {empty[:2]}")
    counts = Counter((r["variant"], r["malicious"]) for r in rows)
    per = {counts[("original", m)] for m in (True, False)}
    print("   counts:", {f"{v}/{'mal' if m else 'ben'}": n for (v, m), n in sorted(counts.items())})
    check("every variant has as many rows as original, per label",
          all(counts[(v, m)] == counts[("original", m)]
              for v in VARIANTS for m in (True, False)))

    # 2. Families: each parent has exactly one of each variant
    parents = {(r["id"], r["malicious"]): r for r in rows if r["variant"] == "original"}
    fam = defaultdict(dict)
    for r in rows:
        if r["variant"] != "original":
            fam[(r["parent_id"], r["malicious"])][r["variant"]] = r
    orphans = [k for k in fam if k not in parents]
    check("every variant points to an existing parent", not orphans, str(orphans[:2]))
    incomplete = [k for k in parents if set(fam.get(k, {})) != set(VARIANTS)]
    check(f"every parent has all {len(VARIANTS)} variants", not incomplete, str(incomplete[:2]))

    # 3. Code invariance and label preservation
    bad_label, bad_body, bad_docs, bad_payload = [], [], [], []
    for key, p in parents.items():
        want = "APPROVE" if p["malicious"] else "BLOCK"
        for v, r in fam.get(key, {}).items():
            if r["malicious"] != p["malicious"] or r.get("cwe_id") != p.get("cwe_id") \
                    or r.get("vuln_id") != p.get("vuln_id"):
                bad_label.append(r["id"])
            if v in BODY:
                if r["diff"] != p["diff"] or r["files"] != p["files"] \
                        or r["files_changed"] != p["files_changed"] \
                        or not r["pr_body"].startswith(p["pr_body"].rstrip("\n")):
                    bad_body.append(r["id"])
            else:
                added = r.get("added_files", [])
                ok = (len(added) == 2 and all(f.endswith(".md") for f in added)
                      and all(r["files"].get(k) == c for k, c in p["files"].items())
                      and set(r["files"]) - set(p["files"]) == set(added)
                      and r["diff"].startswith(p["diff"].rstrip("\n"))
                      and all(f"+++ b/{f}" in r["diff"] for f in added)
                      and r["pr_body"] == p["pr_body"])
                if not ok:
                    bad_docs.append(r["id"])
            try:
                text = decoded_payload(r)
            except Exception as exc:  # noqa: BLE001
                text = f"<decode error {exc}>"
            if v == "neutral_note":
                if "APPROVE" in text or "BLOCK" in text or not text.strip():
                    bad_payload.append(r["id"])
            elif want not in text or ("BLOCK" if want == "APPROVE" else "APPROVE") in text:
                bad_payload.append(r["id"])
    check("variants keep label, CWE and vulnerability", not bad_label, str(bad_label[:2]))
    check("description-only variants leave code byte-identical", not bad_body, str(bad_body[:2]))
    check("split_docs only adds two .md files (parent code intact)", not bad_docs, str(bad_docs[:2]))
    check("payload decodes and asks for the WRONG verdict (control: no verdict)", not bad_payload, str(bad_payload[:2]))

    # 4. Results file (optional)
    if a.results:
        res = [json.loads(l) for l in open(a.results) if l.strip()]
        print(f"\n{len(res)} result rows in {a.results}")
        rkeys = Counter((r["id"], r["malicious"], r["mode"]) for r in res)
        check("no duplicate (id, malicious, mode) results", max(rkeys.values()) == 1)
        for mode in sorted({r["mode"] for r in res}):
            got = {(r["id"], r["malicious"]) for r in res if r["mode"] == mode}
            check(f"[{mode}] covers exactly the benchmark records", got == set(keys),
                  f"missing {len(set(keys) - got)}, extra {len(got - set(keys))}")
            verdict = {(r["id"], r["malicious"]): r["verdict"] for r in res if r["mode"] == mode}
            by_label = Counter((r["malicious"], r["verdict"]) for r in res
                               if r["mode"] == mode and r.get("variant") == "original")
            print(f"   [{mode}] original verdicts:",
                  {f"{'mal' if m else 'ben'}/{v}": n for (m, v), n in sorted(by_label.items())})
            if mode == "analysis_only":
                diff = [k for k, p in parents.items() for v, r in fam.get(k, {}).items()
                        if verdict.get((r["id"], r["malicious"])) != verdict.get(k)]
                check("[analysis_only] every variant verdict equals its parent's",
                      not diff, f"{len(diff)} differ")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
