"""Shared constants and helpers for SEVRA Gitea image handling.

Used by `download_sevra` (pull/save image tars) and `offline_enrich`
(locate local tars + detect Podman/Docker). Live Gitea API enrichment was
removed in favor of offline git-based enrichment.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Fully qualified names avoid Podman short-name TTY prompts.
MALICIOUS_IMAGE = "docker.io/rufimelo/malicious-pr-{cwe}:{version}"
BENIGN_IMAGE = "docker.io/rufimelo/benign-pull-requests:{version}"
DEFAULT_MALICIOUS_VERSION = "deterministic"
DEFAULT_BENIGN_VERSION = "gpt5.2_v2"

LOCAL_IMAGE_ROOT = Path(__file__).resolve().parents[1] / "data" / "SEVRA_images"


def detect_runtime() -> str:
    """Prefer Podman; fall back to Docker. Raises if neither works."""
    for name in ("podman", "docker"):
        path = shutil.which(name)
        if not path:
            continue
        try:
            subprocess.run(
                [path, "version"],
                check=True,
                capture_output=True,
                text=True,
            )
            return path
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError(
        "Neither Podman nor Docker is available. "
        "Install Podman (preferred on this cluster) or Docker, then rerun."
    )


def _local_image_tar(image: str, image_root: Path | None = None) -> Path | None:
    """Return a locally saved image tar path if present."""
    root = image_root or LOCAL_IMAGE_ROOT
    name = image
    if "/" in name:
        name = name.split("/", 1)[1]
    tar_path = root / (name.replace("/", "_").replace(":", "_") + ".tar")
    return tar_path if tar_path.exists() else None
