"""Download SEVRA metadata (Hugging Face) and Gitea container images locally.

Metadata → data/SEVRA/<cwe>/{deterministic,benign/...}/generated_prs.jsonl
Images   → data/SEVRA_images/*.tar  (podman/docker save)

Examples:
  python -m harness.download_sevra                 # metadata + images
  python -m harness.download_sevra --metadata-only
  python -m harness.download_sevra --images-only --cwe cwe89
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import SUPPORTED_CWES, _normalize_cwe_name  # noqa: E402
from harness.gitea_client import (  # noqa: E402
    BENIGN_IMAGE,
    DEFAULT_BENIGN_VERSION,
    DEFAULT_MALICIOUS_VERSION,
    MALICIOUS_IMAGE,
    detect_runtime,
)

DEFAULT_META_ROOT = ROOT / "data" / "SEVRA"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "SEVRA_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cwe",
        action="append",
        default=None,
        help="Limit to one or more CWEs (repeatable). Default: all.",
    )
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=DEFAULT_META_ROOT,
        help="Where to write HF metadata JSONL",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Where to write container image tars",
    )
    parser.add_argument(
        "--malicious-version",
        default=DEFAULT_MALICIOUS_VERSION,
        help="Malicious image/metadata version tag",
    )
    parser.add_argument(
        "--benign-version",
        default=DEFAULT_BENIGN_VERSION,
        help="Benign image/metadata version tag",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only download Hugging Face JSONL",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only pull/save Gitea container images",
    )
    parser.add_argument(
        "--skip-benign",
        action="store_true",
        help="Skip benign metadata and the shared benign image",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local files already exist",
    )
    return parser.parse_args()


def _selected_cwes(args: argparse.Namespace) -> list[str]:
    if not args.cwe:
        return list(SUPPORTED_CWES)
    return [_normalize_cwe_name(name) for name in args.cwe]


def download_metadata(
    *,
    cwes: list[str],
    meta_root: Path,
    malicious_version: str,
    benign_version: str,
    skip_benign: bool,
    force: bool,
) -> int:
    from datasets import load_dataset
    from dataset import HF_DATASET

    written = 0
    for cwe in cwes:
        mal_config = f"{cwe}-{malicious_version}"
        mal_out = meta_root / cwe / malicious_version / "generated_prs.jsonl"
        if mal_out.exists() and not force:
            print(f"  skip metadata {mal_config} (exists)")
        else:
            print(f"  downloading {HF_DATASET} / {mal_config} ...")
            ds = load_dataset(HF_DATASET, mal_config, split="malicious")
            mal_out.parent.mkdir(parents=True, exist_ok=True)
            with mal_out.open("w", encoding="utf-8") as handle:
                for row in ds:
                    handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            print(f"    → {mal_out} ({len(ds)} rows)")
            written += 1

        if skip_benign:
            continue

        ben_config = f"{cwe}-benign"
        # Match HF layout: cwe89/benign/gpt5.2_v2/generated_prs.jsonl
        ben_out = meta_root / cwe / "benign" / benign_version / "generated_prs.jsonl"
        if ben_out.exists() and not force:
            print(f"  skip metadata {ben_config} (exists)")
            continue
        print(f"  downloading {HF_DATASET} / {ben_config} ...")
        ds = load_dataset(HF_DATASET, ben_config, split="benign")
        ben_out.parent.mkdir(parents=True, exist_ok=True)
        with ben_out.open("w", encoding="utf-8") as handle:
            for row in ds:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"    → {ben_out} ({len(ds)} rows)")
        written += 1

    return written


def _image_tar_name(image: str) -> str:
    # docker.io/rufimelo/malicious-pr-cwe89:deterministic
    # → rufimelo_malicious-pr-cwe89_deterministic.tar
    name = image
    if "/" in name:
        name = name.split("/", 1)[1]  # drop registry
    return name.replace("/", "_").replace(":", "_") + ".tar"


def _pull_and_save(
    runtime: str,
    image: str,
    out_dir: Path,
    *,
    force: bool,
    prune_after_save: bool = True,
) -> bool:
    tar_path = out_dir / _image_tar_name(image)
    if tar_path.exists() and not force:
        print(f"  skip image {image} (exists {tar_path.name})")
        return False

    print(f"  pulling {image} ...")
    pull = subprocess.run(
        [runtime, "pull", image],
        capture_output=True,
        text=True,
    )
    if pull.returncode != 0:
        raise RuntimeError(
            f"Failed to pull {image}:\n{pull.stderr or pull.stdout}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving → {tar_path} ...")
    # Stream to file; images are large.
    with tar_path.open("wb") as handle:
        save = subprocess.run(
            [runtime, "save", image],
            stdout=handle,
            stderr=subprocess.PIPE,
        )
    if save.returncode != 0:
        tar_path.unlink(missing_ok=True)
        err = save.stderr.decode("utf-8", errors="replace") if save.stderr else ""
        raise RuntimeError(f"Failed to save {image}:\n{err}")

    size_gb = tar_path.stat().st_size / (1024**3)
    print(f"    ✓ {tar_path.name} ({size_gb:.2f} GiB)")

    if prune_after_save:
        # Rootless graphroot on /tmp is small; free space after each save.
        subprocess.run(
            [runtime, "rmi", "-f", image],
            check=False,
            capture_output=True,
            text=True,
        )
    return True


def download_images(
    *,
    cwes: list[str],
    image_root: Path,
    malicious_version: str,
    benign_version: str,
    skip_benign: bool,
    force: bool,
) -> int:
    runtime = detect_runtime()
    print(f"Using container runtime: {runtime}")
    saved = 0

    for cwe in cwes:
        image = MALICIOUS_IMAGE.format(cwe=cwe, version=malicious_version)
        if _pull_and_save(runtime, image, image_root, force=force):
            saved += 1

    if not skip_benign:
        image = BENIGN_IMAGE.format(version=benign_version)
        if _pull_and_save(runtime, image, image_root, force=force):
            saved += 1

    return saved


def main() -> None:
    args = parse_args()
    if args.metadata_only and args.images_only:
        raise SystemExit("Use only one of --metadata-only / --images-only")

    cwes = _selected_cwes(args)
    do_meta = not args.images_only
    do_images = not args.metadata_only

    if do_meta:
        print(f"==> Metadata → {args.meta_root}")
        n = download_metadata(
            cwes=cwes,
            meta_root=args.meta_root,
            malicious_version=args.malicious_version,
            benign_version=args.benign_version,
            skip_benign=args.skip_benign,
            force=args.force,
        )
        print(f"Metadata files written/updated: {n}")

    if do_images:
        if not shutil.which("podman") and not shutil.which("docker"):
            raise SystemExit(
                "Podman/Docker required for --images. "
                "Re-run with --metadata-only, or install a container runtime."
            )
        print(f"==> Images → {args.image_root}")
        n = download_images(
            cwes=cwes,
            image_root=args.image_root,
            malicious_version=args.malicious_version,
            benign_version=args.benign_version,
            skip_benign=args.skip_benign,
            force=args.force,
        )
        print(f"Image tars written/updated: {n}")
        print(
            "Note: running these images still needs working rootless Podman "
            "(subuid ranges). Local tars let you `podman load -i ...` offline."
        )


if __name__ == "__main__":
    main()
