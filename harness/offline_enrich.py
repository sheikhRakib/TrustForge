"""Offline SEVRA enrichment from local Gitea image tars (no running server).

Works around rootless Podman without /etc/subuid by mounting the image
filesystem via `podman unshare`, copying gitea.db + bare repos, then
reconstructing unified diffs with git.

Example:
  python -m harness.offline_enrich --cwe cwe89 --limit 20
  python -m harness.offline_enrich --cwe cwe89 --hard-split
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    DEFAULT_BENIGN_VERSION,
    DEFAULT_MALICIOUS_VERSION,
    example_from_sevra,
    load_hf_split,
)
from harness.gitea_client import (  # noqa: E402
    BENIGN_IMAGE,
    MALICIOUS_IMAGE,
    _local_image_tar,
    detect_runtime,
)

DEFAULT_OUT = ROOT / "data" / "SEVRA_enriched"
DEFAULT_EXTRACT = Path(os.environ.get("SEVRA_EXTRACT_ROOT", "/tmp/sevra-extract"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwe", required=True, help="CWE id, e.g. cwe89")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT)
    parser.add_argument("--malicious-version", default=DEFAULT_MALICIOUS_VERSION)
    parser.add_argument("--benign-version", default=DEFAULT_BENIGN_VERSION)
    parser.add_argument("--hard-split", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-benign", action="store_true")
    parser.add_argument(
        "--keep-extract",
        action="store_true",
        help="Do not delete extracted db/repos after enrichment",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-copy image data even if extract dir already exists",
    )
    return parser.parse_args()


def _strip_appledouble(root: Path) -> int:
    """Remove macOS AppleDouble junk (._*) that breaks git pack indexes."""
    removed = 0
    for path in root.rglob("._*"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def _ensure_image(runtime: str, image: str) -> None:
    probe = subprocess.run(
        [runtime, "image", "exists", image],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return
    local = _local_image_tar(image)
    if local is None:
        raise RuntimeError(
            f"Image {image} not loaded and no local tar found under data/SEVRA_images/. "
            "Run: python -m harness.download_sevra --images-only"
        )
    print(f"  Loading {local} ...")
    result = subprocess.run(
        [runtime, "load", "-i", str(local)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def extract_image_data(
    *,
    image: str,
    dest: Path,
    force: bool = False,
) -> Path:
    """Mount image via podman unshare and copy gitea.db + bare repos."""
    marker = dest / "gitea.db"
    repos = dest / "repositories"
    if marker.exists() and repos.exists() and not force:
        print(f"  Reusing extract at {dest}")
        return dest

    runtime = detect_runtime()
    if "podman" not in runtime:
        raise RuntimeError("Offline extract requires Podman (image mount + unshare).")

    _ensure_image(runtime, image)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    script = f"""
set -euo pipefail
IMG={image!r}
DEST={str(dest.resolve())!r}
MNT=$(podman image mount "$IMG")
# NFS often rejects preserving container UIDs; copy content only.
cp -R --no-preserve=ownership "$MNT/data/gitea/gitea.db" "$DEST/gitea.db"
mkdir -p "$DEST/repositories"
cp -R --no-preserve=ownership "$MNT/data/git/repositories/." "$DEST/repositories/"
podman image unmount "$IMG" >/dev/null
"""
    print(f"  Extracting {image} → {dest} ...")
    result = subprocess.run(
        ["podman", "unshare", "bash", "-lc", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract {image}:\n{result.stderr or result.stdout}"
        )

    removed = _strip_appledouble(dest)
    if removed:
        print(f"  Removed {removed} AppleDouble (._*) files")
    print(f"  Extract size: {_du(dest)}")
    return dest


def _du(path: Path) -> str:
    result = subprocess.run(
        ["du", "-sh", str(path)],
        capture_output=True,
        text=True,
    )
    return (result.stdout.split() or ["?"])[0]


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        errors="replace",
    )


def _resolve_git_repo(repos_root: Path, owner: str, name: str) -> Path:
    """Resolve bare repo path; Gitea dirs are often lowercased on disk."""
    direct = repos_root / owner / f"{name}.git"
    if direct.exists():
        return direct
    owner_dir = repos_root / owner
    if not owner_dir.exists():
        # Try lowercase owner
        owner_dir = repos_root / owner.lower()
    if owner_dir.exists():
        wanted = f"{name}.git".lower()
        for path in owner_dir.iterdir():
            if path.name.lower() == wanted:
                return path
    raise FileNotFoundError(
        f"Missing bare repo for {owner}/{name} under {repos_root}"
    )


def _sanitize_text(value: str) -> str:
    """Make text safe for single-line JSONL (no raw line breaks / LS / PS)."""
    out = []
    for ch in value:
        o = ord(ch)
        if 0xD800 <= o <= 0xDFFF:
            out.append("\ufffd")
        elif ch in "\u2028\u2029":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _sanitize_obj(obj):
    if isinstance(obj, str):
        return _sanitize_text(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_obj(v) for v in obj]
    return obj


MAX_FILE_BYTES = 200_000


def fetch_pr_offline(
    *,
    db: sqlite3.Connection,
    repos_root: Path,
    repo: str,
    pr_number: int,
) -> dict:
    if "/" not in repo:
        raise ValueError(f"Expected owner/name repo, got {repo!r}")
    owner, name = repo.split("/", 1)
    row = db.execute(
        """
        SELECT pr.head_branch, pr.base_branch, pr.merge_base,
               r.owner_name, r.name AS repo_name,
               i.name AS title, i.content AS body
        FROM pull_request pr
        JOIN repository r ON r.id = pr.base_repo_id
        LEFT JOIN issue i ON i.id = pr.issue_id
        WHERE lower(r.owner_name) = lower(?) AND lower(r.name) = lower(?)
          AND pr."index" = ?
        """,
        (owner, name, int(pr_number)),
    ).fetchone()
    if row is None:
        raise LookupError(f"PR not found in gitea.db: {repo}#{pr_number}")

    head_branch = row["head_branch"] or ""
    merge_base = row["merge_base"] or ""
    gitrepo = _resolve_git_repo(repos_root, row["owner_name"], row["repo_name"])

    head = _git(gitrepo, "rev-parse", f"refs/heads/{head_branch}").strip()
    if not merge_base:
        merge_base = _git(gitrepo, "merge-base", "HEAD", head).strip()

    diff = _sanitize_text(_git(gitrepo, "diff", f"{merge_base}...{head}"))
    paths = [
        line
        for line in _git(gitrepo, "diff", "--name-only", f"{merge_base}...{head}")
        .strip()
        .splitlines()
        if line
    ]

    files: dict[str, str] = {}
    for path in paths:
        try:
            content = _git(gitrepo, "show", f"{head}:{path}")
        except subprocess.CalledProcessError:
            continue
        content = _sanitize_text(content)
        if len(content.encode("utf-8", errors="replace")) > MAX_FILE_BYTES:
            files[path] = (
                content[:MAX_FILE_BYTES]
                + f"\n\n/* truncated: original ~{len(content)} chars; see unified diff */\n"
            )
        else:
            files[path] = content

    return {
        "pr_title": _sanitize_text(row["title"] or ""),
        "pr_body": _sanitize_text(row["body"] or ""),
        "head_branch": head_branch,
        "diff": diff,
        "files_changed": paths,
        "files": files,
    }


def _enrich_rows(
    *,
    rows: list[dict],
    malicious: bool,
    cwe: str,
    extract_dir: Path,
    out_path: Path,
) -> int:
    if not rows:
        print(f"No rows for {out_path}")
        return 0

    db = sqlite3.connect(str(extract_dir / "gitea.db"))
    db.row_factory = sqlite3.Row
    repos_root = extract_dir / "repositories"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            repo = row["repo"]
            pr_number = int(row["pr_number"])
            print(f"[{index}/{len(rows)}] {repo}#{pr_number}...", flush=True)
            try:
                bundle = fetch_pr_offline(
                    db=db,
                    repos_root=repos_root,
                    repo=repo,
                    pr_number=pr_number,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")
                bundle = {
                    "diff": "",
                    "files": {},
                    "files_changed": row.get("files_changed") or [],
                    "pr_title": row.get("pr_title") or "",
                    "pr_body": row.get("pr_body") or "",
                    "error": str(exc),
                }

            example = example_from_sevra(row, malicious=malicious, cwe_dir=cwe)
            if bundle.get("pr_title"):
                example["pr_title"] = bundle["pr_title"]
            if bundle.get("pr_body"):
                example["pr_body"] = bundle["pr_body"]
            if bundle.get("files_changed"):
                example["files_changed"] = bundle["files_changed"]
            example["diff"] = bundle.get("diff") or ""
            example["files"] = bundle.get("files") or {}
            example["head_branch"] = bundle.get("head_branch")
            example["enriched"] = True
            example["enrich_mode"] = "offline"
            if bundle.get("error"):
                example["enrich_error"] = bundle["error"]

            handle.write(
                json.dumps(_sanitize_obj(example), ensure_ascii=False) + "\n"
            )
            written += 1

    db.close()
    print(f"Wrote {written} → {out_path}")
    return written


def main() -> None:
    args = parse_args()
    cwe = args.cwe.lower().replace("-", "")

    print(f"Loading malicious metadata for {cwe}...")
    malicious_rows = load_hf_split(cwe, malicious=True)
    if args.hard_split:
        malicious_rows = [row for row in malicious_rows if row.get("failed_by")]
    if args.limit is not None:
        malicious_rows = malicious_rows[: args.limit]

    mal_image = MALICIOUS_IMAGE.format(cwe=cwe, version=args.malicious_version)
    mal_extract = args.extract_root / f"{cwe}-malicious"
    extract_image_data(
        image=mal_image, dest=mal_extract, force=args.force_extract
    )
    _enrich_rows(
        rows=malicious_rows,
        malicious=True,
        cwe=cwe,
        extract_dir=mal_extract,
        out_path=args.out_root / cwe / "malicious.jsonl",
    )
    if not args.keep_extract:
        shutil.rmtree(mal_extract, ignore_errors=True)

    if args.skip_benign:
        return

    print(f"Loading benign metadata for {cwe}...")
    benign_rows = load_hf_split(cwe, malicious=False)
    if args.limit is not None:
        benign_rows = benign_rows[: args.limit]

    ben_image = BENIGN_IMAGE.format(version=args.benign_version)
    ben_extract = args.extract_root / "benign-shared"
    extract_image_data(
        image=ben_image, dest=ben_extract, force=args.force_extract
    )
    _enrich_rows(
        rows=benign_rows,
        malicious=False,
        cwe=cwe,
        extract_dir=ben_extract,
        out_path=args.out_root / cwe / "benign.jsonl",
    )
    # Keep shared benign extract for other CWEs unless user wants cleanup.
    if not args.keep_extract and args.limit is not None:
        # Limited runs: leave benign extract; full multi-CWE runs can reuse it.
        pass


if __name__ == "__main__":
    main()
