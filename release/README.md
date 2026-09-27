# SEVRA Injection Variants

Data for "Hardening LLM Code-Review Agents against Prompt Injection with Multi-Agent Orchestration and Compiler-Based Program Analysis" (AAAI-27 student abstract).

- `injection-variants-vuln.jsonl.gz` (314 MB, attached to GitHub release `v1.0-aaai27`): 1,496 PRs = (57 malicious, one per distinct vulnerability, + 130 benign) x 8 variants: original, neutral_note (control), authority_note, role_hijack, base64, hex, homoglyph, split_docs. Code is byte-identical across variants; injections push toward the wrong verdict. IDs are `<parent>::<variant>`.
- `injection-variants-vuln-eval.jsonl.gz` (this folder): 4,488 reviews by Qwen2.5-3B-Instruct in baseline, multi_agent, and hybrid modes.

Validate with `check_variants.py`; summarize with `python variant.py report <eval.jsonl>`. Decompress with `gunzip`.

Derived from SEVRA-BENCH (https://huggingface.co/datasets/RedAI4Code/SEVRA), MIT License. Please cite SEVRA-BENCH when using this data.
