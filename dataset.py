"""Dataset loading for TrustForge (SEVRA-BENCH).

Metadata prefers local `data/SEVRA/` (from `python -m harness.download_sevra`),
then falls back to Hugging Face via `datasets.load_dataset`.
Optional enriched records (with real diffs) live under `data/SEVRA_enriched/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
DEFAULT_META = ROOT / "data" / "SEVRA"
DEFAULT_ENRICHED = ROOT / "data" / "SEVRA_enriched"
HF_DATASET = "RedAI4Code/SEVRA"
DEFAULT_MALICIOUS_VERSION = "deterministic"
DEFAULT_BENIGN_VERSION = "gpt5.2_v2"
SUPPORTED_CWES = (
    "cwe22",
    "cwe78",
    "cwe79",
    "cwe89",
    "cwe94",
    "cwe125",
    "cwe352",
    "cwe416",
    "cwe787",
    "cwe862",
)


def _normalize_cwe(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (set, list, tuple)):
        return ",".join(sorted(str(item) for item in value))
    if isinstance(value, dict):
        return ",".join(sorted(str(item) for item in value))
    return str(value)


def _normalize_cwe_name(name: str) -> str:
    return name.lower().replace("-", "")


def example_from_sevra(
    row: dict[str, Any],
    *,
    malicious: bool,
    cwe_dir: str,
) -> dict[str, Any]:
    repo = row.get("repo") or "unknown"
    pr_number = row.get("pr_number")
    vuln_id = row.get("vuln_id") or "unknown"
    example_id = row.get("id") or f"{cwe_dir}:{repo}:pr{pr_number}:{vuln_id}"

    files_changed = list(row.get("files_changed") or [])
    files = dict(row.get("files") or {})
    if not files and row.get("code"):
        path = files_changed[0] if files_changed else "snippet.py"
        files = {path: row["code"]}

    return {
        "id": example_id,
        "malicious": malicious,
        "attack": None if not malicious else row.get("axis3") or row.get("attack"),
        "cwe_id": _normalize_cwe(row.get("cwe_id")),
        "framing": row.get("axis3") or row.get("framing"),
        "pr_title": row.get("pr_title") or "",
        "pr_body": row.get("pr_body") or "",
        "files_changed": files_changed,
        "files": files,
        "diff": row.get("diff") or "",
        "repo": repo,
        "pr_number": pr_number,
        "vuln_id": vuln_id,
        "failed_by": list(row.get("failed_by") or []),
        "enriched": bool(row.get("enriched") or files or row.get("diff")),
        "source": "sevra",
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    # Do not use splitlines(): U+2028/U+2029 inside JSON strings would break rows.
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if line:
                yield json.loads(line)


def local_metadata_path(
    cwe: str,
    *,
    malicious: bool,
    meta_root: Path | None = None,
    malicious_version: str = DEFAULT_MALICIOUS_VERSION,
    benign_version: str = DEFAULT_BENIGN_VERSION,
) -> Path:
    """Path for a locally downloaded SEVRA metadata JSONL file."""
    root = meta_root or DEFAULT_META
    cwe = _normalize_cwe_name(cwe)
    if malicious:
        return root / cwe / malicious_version / "generated_prs.jsonl"
    return root / cwe / "benign" / benign_version / "generated_prs.jsonl"


def load_local_split(
    cwe: str,
    *,
    malicious: bool,
    meta_root: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Load one CWE split from local JSONL, or None if missing."""
    path = local_metadata_path(cwe, malicious=malicious, meta_root=meta_root)
    if not path.exists():
        return None
    return list(iter_jsonl(path))


def load_hf_split(cwe: str, *, malicious: bool) -> list[dict[str, Any]]:
    """Load one SEVRA CWE split (local JSONL preferred, else Hugging Face)."""
    local = load_local_split(cwe, malicious=malicious)
    if local is not None:
        return local

    from datasets import load_dataset

    cwe = _normalize_cwe_name(cwe)
    config = (
        f"{cwe}-{DEFAULT_MALICIOUS_VERSION}" if malicious else f"{cwe}-benign"
    )
    split = "malicious" if malicious else "benign"
    dataset = load_dataset(HF_DATASET, config, split=split)
    return [dict(row) for row in dataset]


def _load_enriched(
    root: Path,
    *,
    cwes: list[str] | None,
    include_benign: bool,
    hard_split_only: bool,
    require_code: bool,
) -> list[dict[str, Any]] | None:
    if not root.exists():
        return None

    cwe_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("cwe")
    )
    if cwes:
        wanted = {_normalize_cwe_name(name) for name in cwes}
        cwe_dirs = [path for path in cwe_dirs if path.name.lower() in wanted]

    if not cwe_dirs:
        return None

    examples: list[dict[str, Any]] = []
    for cwe_dir in cwe_dirs:
        mal_path = cwe_dir / "malicious.jsonl"
        if mal_path.exists():
            for row in iter_jsonl(mal_path):
                example = example_from_sevra(
                    row, malicious=True, cwe_dir=cwe_dir.name
                )
                if hard_split_only and not example.get("failed_by"):
                    continue
                if require_code and not (example["files"] or example["diff"]):
                    continue
                examples.append(example)

        if include_benign:
            ben_path = cwe_dir / "benign.jsonl"
            if ben_path.exists():
                for row in iter_jsonl(ben_path):
                    example = example_from_sevra(
                        row, malicious=False, cwe_dir=cwe_dir.name
                    )
                    if require_code and not (example["files"] or example["diff"]):
                        continue
                    examples.append(example)

    return examples


def _load_hf_metadata(
    *,
    cwes: list[str] | None,
    include_benign: bool,
    hard_split_only: bool,
    require_code: bool,
) -> list[dict[str, Any]]:
    selected = (
        [_normalize_cwe_name(name) for name in cwes] if cwes else list(SUPPORTED_CWES)
    )
    examples: list[dict[str, Any]] = []

    for cwe in selected:
        for row in load_hf_split(cwe, malicious=True):
            example = example_from_sevra(row, malicious=True, cwe_dir=cwe)
            if hard_split_only and not example.get("failed_by"):
                continue
            if require_code and not (example["files"] or example["diff"]):
                continue
            examples.append(example)

        if include_benign:
            for row in load_hf_split(cwe, malicious=False):
                example = example_from_sevra(row, malicious=False, cwe_dir=cwe)
                if require_code and not (example["files"] or example["diff"]):
                    continue
                examples.append(example)

    return examples


def load_benchmark(
    *,
    enriched_root: Path | None = None,
    cwes: list[str] | None = None,
    include_benign: bool = True,
    hard_split_only: bool = False,
    require_code: bool = True,
    prefer_enriched: bool = True,
) -> list[dict[str, Any]]:
    """Load SEVRA examples. Defaults to enriched local diffs (metadata + code)."""
    enriched_root = enriched_root or DEFAULT_ENRICHED
    examples: list[dict[str, Any]] | None = None

    if prefer_enriched:
        examples = _load_enriched(
            enriched_root,
            cwes=cwes,
            include_benign=include_benign,
            hard_split_only=hard_split_only,
            require_code=require_code,
        )

    if not examples:
        if prefer_enriched and require_code:
            raise FileNotFoundError(
                f"No enriched examples with diffs under {enriched_root}. "
                "Run: python -m harness.offline_enrich --cwe <cwe>"
            )
        examples = _load_hf_metadata(
            cwes=cwes,
            include_benign=include_benign,
            hard_split_only=hard_split_only,
            require_code=require_code,
        )

    if require_code:
        incomplete = [
            item["id"]
            for item in examples
            if not ((item.get("diff") or "").strip() and (item.get("pr_title") or "").strip())
        ]
        if incomplete:
            preview = ", ".join(incomplete[:5])
            more = f" (+{len(incomplete) - 5} more)" if len(incomplete) > 5 else ""
            raise ValueError(
                "Benchmark examples missing title and/or diff: "
                f"{preview}{more}"
            )

    return examples


def format_pr_for_review(example: dict[str, Any]) -> str:
    """Render a PR-shaped example as text for LLM agents.

    Always includes metadata (title, body, paths) and a unified diff.
    Raises if either side is missing so evals cannot silently go metadata-only.
    """
    title = (example.get("pr_title") or "").strip()
    body = (example.get("pr_body") or "").strip()
    diff = (example.get("diff") or "").strip()
    files_changed = list(example.get("files_changed") or [])
    files = dict(example.get("files") or {})

    missing: list[str] = []
    if not title:
        missing.append("pr_title")
    if not body:
        missing.append("pr_body")
    if not diff:
        missing.append("diff")
    if missing:
        raise ValueError(
            f"Example {example.get('id')!r} missing required fields for review: "
            f"{', '.join(missing)}. Run `python -m harness.offline_enrich --cwe <cwe>` "
            "and ensure data/SEVRA_enriched is loaded."
        )

    parts = [
        f"### Pull Request Title\n{title}",
        f"### Pull Request Description\n{body}",
        "### Files Changed\n"
        + ("\n".join(f"- {path}" for path in files_changed) or "(none listed)"),
        f"### Unified Diff\n```diff\n{diff}\n```",
    ]

    if files:
        file_blocks = [
            f"#### {path}\n```\n{content}\n```" for path, content in files.items()
        ]
        parts.append("### File Contents (PR head)\n" + "\n\n".join(file_blocks))

    return "\n\n".join(parts)
