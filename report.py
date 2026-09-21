"""Export evaluation metrics and the paper's ASR-by-variant figure."""

import argparse
import csv
import json
import re
from pathlib import Path
from main import is_slurm_log_path, summarize
from dataset import iter_jsonl

DEFAULT_REPORT_DIR = Path("output/report")


def load_results(path, *, allow_partial=False):
    rows = list(iter_jsonl(path))
    keys = [(r["id"], r["malicious"], r["mode"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate example/mode records in results")
    modes = sorted({r["mode"] for r in rows})
    groups = [
        {(r["id"], r["malicious"]) for r in rows if r["mode"] == m} for m in modes
    ]
    aligned = bool(rows) and all(group == groups[0] for group in groups)
    if not aligned and not allow_partial:
        raise ValueError(
            "Results are empty or observed modes cover different examples. "
            "Use --allow-partial only for diagnostic reports."
        )
    return rows, aligned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    if is_slurm_log_path(a.output_dir):
        p.error("logs/ is reserved for Slurm/runtime logs; use output/ for reports")
    try:
        rows, complete = load_results(a.results, allow_partial=a.allow_partial)
    except ValueError as exc:
        p.error(str(exc))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    for mode in sorted({r["mode"] for r in rows}):
        for source in sorted({r.get("source", "sevra") for r in rows}):
            for variant in sorted({r.get("variant", "original") for r in rows}):
                selected = [
                    r
                    for r in rows
                    if r["mode"] == mode
                    and r.get("source", "sevra") == source
                    and r.get("variant", "original") == variant
                ]
                if selected:
                    groups.append(
                        dict(
                            source=source, variant=variant, **summarize(selected, mode)
                        )
                    )
    if not groups:
        raise SystemExit("No results")
    (a.output_dir / "report_status.json").write_text(
        json.dumps(
            {
                "results": str(a.results.resolve()),
                "records": len(rows),
                "mode_alignment_verified": complete,
                "scope": "all observed modes contain the same examples"
                if complete
                else "observed modes contain different examples",
            },
            indent=2,
        )
        + "\n"
    )
    with (a.output_dir / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for source in sorted({r["source"] for r in groups}):
        data = [r for r in groups if r["source"] == source]
        variants = sorted({r["variant"] for r in data})
        modes = sorted({r["name"] for r in data})
        fig, ax = plt.subplots(figsize=(max(6, len(variants) * 1.3), 4))
        width = 0.8 / len(modes)
        for j, mode in enumerate(modes):
            for i, variant in enumerate(variants):
                record = next(
                    (r for r in data if r["name"] == mode and r["variant"] == variant),
                    None,
                )
                if record and record["attack_success"] is not None:
                    x = i - 0.4 + width / 2 + j * width
                    ax.bar(
                        x,
                        100 * record["attack_success"],
                        width,
                        label=mode if i == 0 else None,
                        color=f"C{j}",
                    )
                    ax.text(
                        x,
                        100 * record["attack_success"] + 1,
                        f"n={record['malicious_n']}",
                        ha="center",
                        fontsize=7,
                    )
        ax.set_xticks(range(len(variants)), variants, rotation=25, ha="right")
        ax.set_ylabel("Malicious approval rate (%)")
        ax.set_ylim(0, 112)
        ax.set_title(
            f"Observed ASR by variant — {source}"
            + ("" if complete else " (partial/unverified)")
        )
        ax.legend()
        fig.tight_layout()
        safe_source = re.sub(r"[^A-Za-z0-9_-]", "_", source)
        fig.savefig(a.output_dir / f"asr_{safe_source}.svg")
        plt.close(fig)
    print(f"Wrote metrics.csv and ASR figure(s) to {a.output_dir}")


if __name__ == "__main__":
    main()
