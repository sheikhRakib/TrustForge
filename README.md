# TrustForge

TrustForge evaluates LLM PR-review agents on **SEVRA-BENCH**, with a focus on
hardening against prompt injection / social engineering in malicious PRs.

Hugging Face metadata alone is not enough for a real review: the agent needs
the PR title/body **and** a unified code diff. Real diffs ship in companion
Gitea container images from `malicious-pr-bench`:

- Malicious: `docker.io/rufimelo/malicious-pr-<cwe>:deterministic`
- Benign: `docker.io/rufimelo/benign-pull-requests:gpt5.2_v2`

We do **not** rewrite the original Hugging Face JSONL. Enrichment writes
**new** files under `data/SEVRA_enriched/`.

## Reproduce from scratch

End-to-end path: clone → conda env → download SEVRA → offline enrich → eval.

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Linux + git | Cluster or workstation |
| Conda / Miniconda | Env defined in `environment.yml` |
| GPU + CUDA | Default model is `Qwen/Qwen3-Coder-30B-A3B-Instruct` (multi-GPU recommended; scripts request 2× GPU, ~90 GB RAM) |
| Podman (or Docker) | Needed to load/mount Gitea image tars for enrichment |
| `git` CLI | Used inside enrichment to build unified diffs |
| Disk | ~9 GB image tars + enriched JSONL + model weights (tens of GB) |
| Hugging Face access | `datasets` + model download; run `huggingface-cli login` if gated |

Offline enrichment mounts image layers via `podman unshare` and does **not**
need `/etc/subuid`. On NFS home, still point Podman `graphroot` at local disk
(see [Rootless Podman](#rootless-podman-on-this-cluster)).

### 1. Clone and create the environment

```bash
git clone <this-repo-url> TrustForge
cd TrustForge

conda env create -f environment.yml
conda activate trustforge
```

To refresh an existing env after dependency changes:

```bash
conda env update -f environment.yml --prune
conda activate trustforge
```

### 2. Download SEVRA metadata and Gitea images

```bash
# Full download (all CWEs; large)
python -m harness.download_sevra

# Smaller smoke path
python -m harness.download_sevra --metadata-only
python -m harness.download_sevra --images-only --cwe cwe89
```

Expected artifacts:

```
data/SEVRA/<cwe>/deterministic/generated_prs.jsonl
data/SEVRA/<cwe>/benign/gpt5.2_v2/generated_prs.jsonl
data/SEVRA_images/rufimelo_malicious-pr-<cwe>_deterministic.tar
data/SEVRA_images/rufimelo_benign-pull-requests_gpt5.2_v2.tar
```

`dataset.py` prefers local `data/SEVRA/` over Hugging Face when present.
Enrichment prefers `podman load` from local tars when online pull is unavailable.

On NFS home, configure Podman storage under `/tmp` before enrichment
(see [Rootless Podman](#rootless-podman-on-this-cluster)).

### 3. Enrich with real PR diffs

```bash
# Recommended offline path (no live Gitea / no subuids)
python -m harness.offline_enrich --cwe cwe89 --hard-split

# Smoke test with a small slice
python -m harness.offline_enrich --cwe cwe89 --hard-split --limit 20

# Optional: keep extracted db/repos, re-copy from image, skip benign
#   --keep-extract  --force-extract  --skip-benign
```

Writes:

```
data/SEVRA_enriched/cwe89/malicious.jsonl
data/SEVRA_enriched/cwe89/benign.jsonl
```

Repeat `--cwe` for each CWE you want to evaluate. Eval loads
`data/SEVRA_enriched/` by default and refuses examples without title+diff
unless you pass `--allow-metadata-only` (not recommended).

### 4. Run evaluation

**Interactive / local GPU node:**

```bash
python main.py --cwe cwe89 --hard-split --limit 20
python main.py --cwe cwe89 --hard-split --modes baseline,multi_agent,hybrid
```

**Useful flags:**

| Flag | Meaning |
| --- | --- |
| `--cwe cwe89` | Filter CWE (repeatable) |
| `--hard-split` | Keep samples that fooled ≥1 baseline model |
| `--limit N` | Cap number of examples |
| `--modes baseline,multi_agent,hybrid` | Which defenses to run |
| `--model <hf-id>` | Override default Qwen coder model |
| `--allow-metadata-only` | Escape hatch; skip diff requirement |

Metrics printed: **ASR** (malicious → APPROVE), **clean approval**, **FPR**.

**SLURM (this cluster’s helpers):**

```bash
# Interactive allocation + run
./run_srun.sh --cwe cwe89 --hard-split --limit 20

# Batch job
sbatch --export=ALL,EXTRA_ARGS="--cwe cwe89 --hard-split --limit 20" run.slurm
```

Logs go under `logs/`. Adjust partition / GPU / mem in `run_srun.sh` and
`run.slurm` if your site differs.

### 5. Minimal smoke checklist

```bash
conda activate trustforge
python -m harness.download_sevra --metadata-only
python -m harness.download_sevra --images-only --cwe cwe89
python -m harness.offline_enrich --cwe cwe89 --hard-split --limit 5
python main.py --cwe cwe89 --hard-split --limit 5 --modes baseline
```

If enrichment fails on image mount, fix Podman `graphroot` (local disk, not
NFS) and confirm `podman image exists` / `podman load -i data/SEVRA_images/…`.
If eval OOM’s, reduce `--limit`, use fewer `--modes`, or request more GPUs.

## Data preprocessing pipeline

```
RedAI4Code/SEVRA (HF)          Gitea container images (Docker Hub)
        │                                  │
        ▼                                  ▼
 data/SEVRA/                     data/SEVRA_images/*.tar
 (metadata JSONL)                (docker-save archives)
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              harness/offline_enrich.py
                       │
                       ▼
              data/SEVRA_enriched/
              <cwe>/{malicious,benign}.jsonl
                       │
                       ▼
              dataset.load_benchmark() → main.py
```

| Stage | Command | Output |
| --- | --- | --- |
| 1. Download metadata + images | `python -m harness.download_sevra` | `data/SEVRA/`, `data/SEVRA_images/` |
| 2. Enrich with real diffs | `python -m harness.offline_enrich --cwe …` | `data/SEVRA_enriched/` |
| 3. Evaluate | `python main.py …` | loads enriched JSONL by default |

### Stage 1 — Download

#### `data/SEVRA/` (Hugging Face metadata)

Source: [`RedAI4Code/SEVRA`](https://huggingface.co/datasets/RedAI4Code/SEVRA).

```
data/SEVRA/<cwe>/deterministic/generated_prs.jsonl   # malicious
data/SEVRA/<cwe>/benign/gpt5.2_v2/generated_prs.jsonl
```

Typical fields: `repo`, `pr_number`, `pr_title`, `pr_body`, `files_changed`,
`cwe_id`, `vuln_id`, attack axes (`axis1`–`axis3`), `failed_by`, etc.

There is **no** unified diff and **no** full file contents here. Benign HF
rows often have an empty `pr_body`; titles/bodies are filled later from Gitea.

#### `data/SEVRA_images/` (Gitea image tars)

Official `malicious-pr-bench` images, saved as Podman/Docker archives:

| File | Image |
| --- | --- |
| `rufimelo_malicious-pr-<cwe>_deterministic.tar` | `docker.io/rufimelo/malicious-pr-<cwe>:deterministic` |
| `rufimelo_benign-pull-requests_gpt5.2_v2.tar` | `docker.io/rufimelo/benign-pull-requests:gpt5.2_v2` |

Each `.tar` is a **docker-save** bundle (`manifest.json`, config, layered
filesystem tars)—not a folder of `.diff` files.

Inside the top filesystem layer (Gitea data volume):

```
data/gitea/
  gitea.db                 # SQLite: PR index, merge_base, issue title/body
  conf/app.ini
  …

data/git/repositories/
  <owner>/<repo>.git/      # bare git repos (HEAD, refs/, objects/, …)
```

PR metadata lives in `gitea.db`. Code history lives in the bare repos.
Unified diffs are **computed** later with `git`, not shipped as separate
patch files.

### Stage 2 — Offline enrichment

Rootless Podman on this cluster often cannot *run* the Gitea container
(missing `/etc/subuid`). Enrichment therefore mounts the image with
`podman unshare`, copies out what we need, and reconstructs diffs offline.

What it does per PR:

1. Look up the PR in `gitea.db` (`merge_base`, head branch, title, body).
2. Resolve the bare repo under `repositories/`.
3. Run `git diff <merge_base>...<head>` → unified patch.
4. `git show <head>:<path>` for each changed file → post-change contents
   (truncated above ~200KB).
5. Merge that bundle onto the HF metadata row and write one JSONL line.

#### `data/SEVRA_enriched/` (eval input)

```
data/SEVRA_enriched/<cwe>/malicious.jsonl
data/SEVRA_enriched/<cwe>/benign.jsonl
```

One JSON object per line. Fields added or filled beyond HF metadata:

| Field | Meaning |
| --- | --- |
| `diff` | Full unified diff for the PR |
| `files` | Map of path → file content at head |
| `files_changed` | Paths from the diff |
| `pr_title` / `pr_body` | Prefer Gitea issue text when present |
| `head_branch` | PR head branch name |
| `enriched` | `true` |
| `enrich_mode` | `"offline"` |

Light transforms only: UTF-8 sanitization and large-file truncation.
Attack narratives and vulnerability labels from HF are not rewritten.

Extracts are written under `SEVRA_EXTRACT_ROOT` (default `/tmp/sevra-extract`,
or `data/SEVRA_extract` if you point it there). Malicious extracts are deleted
after enrichment unless `--keep-extract` is set; they are not required once
`data/SEVRA_enriched/` exists.

### Stage 3 — Loading at eval time

`dataset.load_benchmark()` prefers `data/SEVRA_enriched/` and, by default,
**requires** a non-empty title and diff (`require_code=True`).

The review prompt (`format_pr_for_review`) includes PR title, body, changed
paths, unified diff, and file contents—not the diff alone.

### Design notes

- **Why images?** SEVRA’s HF release is metadata-centric; real patches and
  repos ship in the companion Gitea images from the SEVRA / malicious-pr-bench
  authors.
- **Why offline enrich?** Avoids needing a live Gitea server when rootless
  containers cannot start.
- **Why a separate enriched tree?** Keeps downloaded HF JSONL and image tars
  immutable; enrichment is reproducible from those artifacts.

## Rootless Podman on this cluster

1. **Local storage** (NFS home breaks volume xattrs). Put this in
   `~/.config/containers/storage.conf`:

```
[storage]
driver = "overlay"
runroot = "/run/user/<UID>/containers"
graphroot = "/tmp/podman-<USER>/storage"

[storage.options]
ignore_chown_errors = "true"

[storage.options.overlay]
force_mask = "700"
mount_program = "/usr/bin/fuse-overlayfs"
mountopt = "nodev,fsync=0,ignore_chown_errors"
```

Offline enrichment only needs image mount via `podman unshare` (no subuids).

## Related scripts

| Module | Role |
| --- | --- |
| `harness/download_sevra.py` | HF JSONL + image tar download |
| `harness/offline_enrich.py` | Mount image → db + repos → enriched JSONL |
| `harness/gitea_client.py` | Image names, local tar paths, runtime detection |
| `dataset.py` | Load metadata / enriched splits for eval |
| `main.py` | Eval entrypoint |
| `agent.py` / `analysis.py` / `model.py` | Review agents and Lane B analysis |
| `run_srun.sh` / `run.slurm` | Cluster launch helpers |
