# TrustForge

TrustForge evaluates defenses against prompt injection in pull-request review.
It compares a single-model reviewer, role-separated reviewers, and a hybrid with
program analysis. The current default model is
`Qwen/Qwen2.5-3B-Instruct` (matches the paper's Qwen2.5 family). Report the
exact Hugging Face ID; results are not comparable to earlier 30B coder runs.

## Start with the data already downloaded

This workspace already contains the reproduction inputs. Do **not** download or
enrich them again to run an evaluation.

| Local artifact | Purpose | Approximate size |
| --- | --- | --- |
| `data/SEVRA/` | Original SEVRA metadata JSONL across ten CWEs | 5.5 MB |
| `data/SEVRA_images/` | Downloaded malicious and benign Gitea image archives | 8.6 GB |
| `data/SEVRA_enriched/` | PR titles, bodies, unified diffs, changed-file contents | 474 MB |

The current enriched dataset has **2,597 examples: 2,250 malicious and 347 benign**.
These data directories and run logs are excluded from Git; a fresh clone does
not contain them. Their existence in this workspace is separate from Git tracking.

```bash
conda activate trustforge
# After pulling these implementation changes, update the environment once:
conda env update -f environment.yml

# Validate existing data and options without loading model weights:
python main.py --dry-run

# The fixed Slurm script runs the four selected CWEs with all three systems:
sbatch run.slurm

# On an allocated GPU node, evaluate baseline through the shared runner:
python main.py --cwe cwe78 --cwe cwe79 --cwe cwe89 --cwe cwe94 \
  --modes baseline --output output/baseline-from-main.jsonl
```

The launchers request one GPU, 32 GB host RAM, and 12 hours (enough for the
default 3B model). A full run may need more than one allocation. Resubmit the
**same command** to resume completed example/mode records. Use a new output path
after changing code, data, model, options, or ablations to avoid mixing results.
`logs/` is reserved for Slurm stdout/stderr and GPU telemetry. Model responses,
summaries, reports, and figures belong
under `output/`. Create `logs/` before submitting on a fresh clone because Slurm
opens its log files before the script starts.

For an interactive allocation, the launcher forwards options to `main.py`:

```bash
./run_srun.sh --limit 4 --output output/interactive.jsonl
```

On an already allocated GPU node, use `python main.py` with the same arguments.
The default is one model sharded across the visible GPUs. CUDA is required and
CPU/disk model offload is rejected rather than silently making inference slow.
Set `TRUSTFORGE_CONDA_SH` if Conda is not at `~/miniconda3/etc/profile.d/conda.sh`.
Adjust the partition, QoS, GPU count, and memory in the launchers for other sites.

## Evaluation options and outputs

| Option | Implemented behavior |
| --- | --- |
| `--cwe cwe89` | Repeatable CWE filter for local SEVRA |
| `--hard-split` | Keep malicious examples with an upstream `failed_by` entry; keep benign controls |
| `--limit N --seed 42` | Deterministic sampling balanced across labels when available |
| `--modes baseline,multi_agent,hybrid` | Select any combination of systems; the default is all three. `analysis_only` also works without a GPU |
| `--benchmark PATH` | Evaluate an existing benchmark JSONL; requires PR metadata/diff; absent head files cause auditor abstention |
| `--max-input-tokens 32768` | Per-call input budget; also reserves output space in the model context |
| `--model ID` | Hugging Face model override; model must support the configured SDPA attention backend |
| `--disable stripping,grounding` | Component ablations; see below |
| `--output PATH` | Append-only results with checkpoint resume |

For an output `output/full.jsonl`, the runner writes:

- `full.jsonl`: verdicts, model responses, labels, and example metadata.
- `full.runtime.json`: model revision, GPU names, CUDA version (LLM runs).
- `full.summary.json`: overall and per-variant/framing/CWE/source metrics on completion.

ASR is the fraction of malicious examples receiving APPROVE. Clean approval is
the fraction of benign examples receiving APPROVE. FPR counts benign COMMENT
or BLOCK decisions. UNKNOWN is reported separately and never scored as correct.
A metric with no examples in its denominator is `null`, not zero. For injected
benign controls, FPR measures the attempted denial of approval. These are decision
metrics; a low ASR alone does not establish useful review quality.

```bash
python report.py output/full.jsonl --output-dir output/full-report
```

This exports metrics CSV, report status, and SVG ASR figures by variant,
separating SEVRA and synthetic fixtures. By default it requires every observed
mode to contain the same examples. For diagnostic reports with uneven observed
modes, add `--allow-partial`; their figures are labeled partial/unverified. The
report cannot infer whether an entirely absent mode or record was expected.

## What is implemented

- Scanner triage prioritizes files; the injection detector flags reviewer
  manipulation. These are separate model calls using the same weights.
- The auditor receives changed diff hunks and numbered, syntax-stripped
  current source lines within five lines of each change. It compares removed
  and added code before returning a short verdict plus an optional line ID.
  The code constructs citations from
  visible lines; the model does not write citation paths or quotes. Tree-sitter
  handles Python, JavaScript/TypeScript, C/C++, PHP,
  Java, Go, Ruby, Rust, C#, Bash, Swift, Kotlin, and Scala. Python docstrings
  are removed using Python AST positions. Literal contents are preserved.
- This diff-scoped review applies to source files and lockfiles alike. If a
  matching hunk is unavailable, the auditor reviews the full supplied source
  and reports a warning.
- Grounding validates the exact file, original line number, and nonempty quoted
  source excerpt, including whether it was visible in the auditor call. It
  checks evidence existence, not the correctness of the model's reasoning.
- Python analysis propagates taint through assignments, supported expressions,
  and resolved local function calls. SQL taint checks distinguish the query from
  bound parameters. Imported Flask request aliases are recognized; standalone
  top-level parameters named `request` use an explicitly reported HTTP-request
  assumption. Reassignment clears that identity. It resolves absolute/relative imports and
  aliases among supplied Python modules. Optional repository context includes
  unchanged Python files; changed files take precedence.
- Z3 checks supported Python branch constraints and produces concrete modeled
  input values for tainted sink paths. Bounds: 3,000 interpreter steps, 64 live
  states, call depth 4, and a 100 ms solver timeout. Classes/methods, nested
  functions, loops, dynamic dispatch, complex objects, and unsupported expressions
  are reported as coverage limits.
  Witnesses describe a modeled reachable sink, not proof of exploitability.
- Multilingual sink/taint heuristics (tree-sitter) cover PHP, JavaScript/TypeScript,
  C/C++, Go, Java, Ruby, and Rust: known dangerous calls are inventoried, and
  advisory ``taint`` findings fire when a known source and sink co-occur in the
  same function or when a high-risk sink (eval/system/exec-style) appears.
  These are heuristics, not symbolic proofs.
- Multilingual diff rules identify potential changes to sinks and protection
  code. Syntax comments and literal text are masked before matching; executable
  string interpolation is retained. SQL diff rules inspect API argument
  construction rather than keywords inside strings. These are advisory
  heuristics, not compiler proofs. Ground-truth CWE labels do not select rules.
- The multi-agent aggregator blocks source-grounded auditor defects and comments
  on auditor concerns. Narrative suspicion alone does not reject otherwise
  approved code. Safe approvals do not require a nonexistent vulnerability
  citation. Missing head files (including deletion-only PRs) and malformed auditor decisions remain UNKNOWN.
- Hybrid uses the scanner, injection detector, auditor, and program analysis.
  A separate final model review considers their bounded signals alongside the
  PR diff and makes its own verdict; it does not inherit the multi-agent verdict.
  Component signals are advisory, so the final reviewer can approve a fix that
  the auditor flagged or reject a defect the auditor missed.

There is no claim of complete, injection-immune verification: full symbolic and
import-aware analysis remains Python-specific; other supported languages use
sink/taint heuristics plus syntax stripping and diff rules. Parser errors and
unsupported syntax are exposed in diagnostics. Imported modules absent from the
supplied context cannot be resolved. Coverage warnings are diagnostic: they do
not automatically override an auditor approval. `analysis_only` returns UNKNOWN
when it has no advisory finding; absence of a finding is not treated as a
clean-code proof.

## Input coverage and GPU performance

The model's default input budget is **32,768 tokens**, matching the Qwen2.5-3B
context window. Earlier 30B runs used 8,192 because a 49,152-token budget
exhausted memory on two A100 40 GB GPUs; the smaller 3B model makes the full
window practical. Oversized content is still reviewed in chunks when it exceeds
the budget; system instructions and chat formatting are kept for every call.
PR narrative/diff prompts no longer silently discard their tails. Auditor
chunks cover changed hunks and nearby current source. Files without a matching
hunk fall back to full supplied source, with a warning.
Extremely long lines may cross chunk boundaries, limiting evidence citation
for those line fragments.

Previously enriched files may already contain the old `/* truncated: ... */`
marker from the former 200,000-character enrichment cap. Increasing model context
cannot recover that missing source. New enrichment preserves full source. Rebuild
into a separate output directory from the downloaded images if you need to remove
old enrichment truncations or include unchanged repository Python files.

Inference uses GPU SDPA attention, inference mode, and the KV cache. The existing
Transformers generation implementation computes only needed output logits.
When both modes run on the same PR, hybrid reuses the scanner, injection, and
auditor calls, then adds its own final model calls. It can reach a different
verdict from multi-agent.
Small source files share auditor calls.
The Slurm log prints elapsed time and model-call count for each review. JSONL
results omit per-call timing, token counts, GPU memory, and OOM retry details;
OOM recovery still retries the complete review with a smaller input budget.

Full source coverage can require more work than earlier truncated runs; GPU use
alone is not a guarantee of high throughput on every input.

For a live terminal view of GPU utilization on a running job:

```bash
./monitor_gpus.sh JOBID
```

After CUDA OOM, the runner clears the cached review and retries the complete
input at half the chunk budget, up to four retries and a minimum of 512 tokens.
Each mode/example starts at the configured budget so resume order does not
inherit another example's reduction. Exhausted retries stop the run and leave
the failed result pending; no partial review is scored as complete.

OOM is a per-call memory issue: reducing `--limit` does not make a long prompt
fit. Lower `--max-input-tokens` to create smaller chunks, or request more GPU
memory. Larger models may need a lower default budget and more GPUs than the
current 3B launchers.

## Ablations

```bash
# Same dataset/sample seed, separate output for each ablation:
./run_srun.sh --disable stripping --output output/no-stripping.jsonl
./run_srun.sh --disable grounding --output output/no-grounding.jsonl
./run_srun.sh --disable symbolic --output output/no-symbolic.jsonl
```

This repository no longer generates narrative variants or synthetic fixtures.
If you already have a benchmark JSONL containing them, `main.py --benchmark`
can still evaluate it. Ablation names are `scanner`,
`injection`, `stripping`, `grounding`, `taint`, `symbolic`, `cross_file`, and `diff`.
`symbolic` disables constraint solving and witnesses while retaining taint;
`taint` disables taint and its dependent symbolic witnesses while retaining the
structural sink inventory. Both retain the shared AST traversal. The code and
reporting tools do not establish paper results until the corresponding runs finish.
This repository performs inference/evaluation; it does not train or fine-tune the
reviewer.

## Rebuild data on a fresh machine

Only use this section if the local artifacts described above are absent or if
explicitly rebuilding enriched source coverage.

```bash
conda env create -f environment.yml
conda activate trustforge
mkdir -p logs
python -m harness.download_sevra
python -m harness.offline_enrich --cwe cwe89 --include-repository-source
```

Repeat offline enrichment separately for each desired CWE; `--cwe` accepts one
value there. The downloader accepts repeatable CWE filters and
`--metadata-only` / `--images-only`. Downloaded metadata comes from
[RedAI4Code/SEVRA](https://huggingface.co/datasets/RedAI4Code/SEVRA); code history
comes from the companion Gitea image archives.

Offline enrichment requires **Podman**, working image mounting via
`podman unshare`, and Git. Docker is supported for downloading archives, but not
by the offline mount implementation. Rootless configuration depends on the
host; successful `podman unshare` is required, not guaranteed by this repository.
Use local container storage on hosts where NFS does not support its filesystem
operations.

```bash
# Reuse already downloaded images, preserving the current enriched dataset:
python -m harness.offline_enrich --cwe cwe89 --include-repository-source \
  --out-root data/SEVRA_enriched_full
```

Enrichment derives each diff from the PR merge base and head commit and reads
source with `git show`. It does not execute submitted code. Existing output is
protected unless `--force-output` is explicit. New splits are written through a
temporary file. `--limit` and `--hard-split` change the enriched dataset and
should use a separate `--out-root`. The evaluator's default enriched root remains
`data/SEVRA_enriched`; concatenate chosen rebuilt JSONL splits into a new file and
pass it via `--benchmark` to evaluate an alternate root.

## Verification

```bash
python -m unittest discover -s tests -v
python main.py --dry-run
bash -n run.slurm run_srun.sh monitor_gpus.sh
```

Tests cover source stripping, literal-text false positives, SQL parameterization,
request-source assumptions, grounding, taint, satisfiable/unsatisfiable paths,
import resolution, multilingual sink/taint heuristics, label-independent analysis,
invalid verdict scoring, chunk coverage, OOM recovery,
report completion checks, and reuse and timing of agent calls.
