"""Export evaluation metrics and the SEVRA attack-type ASR chart."""

import argparse
import csv
import json
from pathlib import Path
from main import is_slurm_log_path, summarize
from dataset import iter_jsonl

DEFAULT_REPORT_DIR = Path("output/report")


def write_attack_type_figure(rows, output_dir, *, complete):
    """Plot baseline and hybrid ASR by original SEVRA attack/framing label."""
    selected = [
        row for row in rows
        if row["malicious"]
        and row.get("source", "sevra") == "sevra"
        and row.get("variant", "original") == "original"
        and (row.get("attack") or row.get("framing"))
        and row["mode"] in ("baseline", "hybrid")
    ]
    if {row["mode"] for row in selected} != {"baseline", "hybrid"}:
        return False

    labels = {}
    counts = {}
    for row in selected:
        label = row.get("attack") or row["framing"]
        key = (row["id"], row["malicious"])
        if key in labels and labels[key] != label:
            raise ValueError(f"Inconsistent attack type across modes for {row['id']}")
        labels[key] = label
        bucket = counts.setdefault((label, row["mode"]), [0, 0])
        bucket[0] += 1
        bucket[1] += row["verdict"] == "APPROVE"

    modes = ("baseline", "hybrid")
    rank_mode = "hybrid"
    attack_types = sorted(
        {label for label, _ in counts},
        key=lambda label: (
            counts.get((label, rank_mode), [0, 0])[1]
            / max(1, counts.get((label, rank_mode), [0, 0])[0]),
            label,
        ),
    )
    summary = []
    for label in attack_types:
        entry = {"attack_type": label}
        for mode in modes:
            n, approved = counts.get((label, mode), [0, 0])
            entry.update({
                f"{mode}_malicious_n": n,
                f"{mode}_approved": approved,
                f"{mode}_asr": approved / n if n else "",
            })
        summary.append(entry)

    stem = output_dir / "attack-type-asr"
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.55, max(2.8, 0.31 * len(summary) + 0.5)))
    styles = {
        "baseline": ("#c15d27", "s", "Baseline"),
        "hybrid": ("#176b9a", "o", "Hybrid"),
    }
    offsets = [0] if len(modes) == 1 else [
        -0.15 + 0.30 * i / (len(modes) - 1) for i in range(len(modes))
    ]
    for index, mode in enumerate(modes):
        color, marker, label = styles.get(mode, (f"C{index}", "D", mode))
        points = [
            (100 * counts[(row["attack_type"], mode)][1]
             / counts[(row["attack_type"], mode)][0], i + offsets[index])
            for i, row in enumerate(summary)
            if (row["attack_type"], mode) in counts
        ]
        if points:
            ax.scatter(*zip(*points), s=22, marker=marker, color=color,
                       label=label, zorder=3)
    ax.set_yticks(range(len(summary)),
                  [row["attack_type"].replace("_", " ") for row in summary],
                  fontsize=6.7)
    ax.set_xlim(0, 105)
    ax.set_xticks(range(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)],
                  fontsize=7)
    ax.set_xlabel("Attack success rate", fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e4e7eb", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#aab2bb")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(modes),
              frameon=False, fontsize=6.4, handletextpad=0.2, columnspacing=0.7)
    denominators = {n for n, _ in counts.values()}
    count_note = (f"n={denominators.pop()} malicious PRs/type" if len(denominators) == 1
                  else "sample counts in CSV")
    fig.text(0.5, 0.012,
             f"SEVRA attack labels; {count_note}; original variants"
             + ("; partial results" if not complete else ""),
             ha="center", fontsize=6.2, color="#59636e")
    fig.subplots_adjust(left=0.47, right=0.97, top=0.94, bottom=0.10)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


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
    attack_figure = write_attack_type_figure(rows, a.output_dir, complete=complete)
    if attack_figure:
        print(f"Wrote attack-type ASR figure and CSV to {a.output_dir}")
    print(f"Wrote metrics.csv and report status to {a.output_dir}")


if __name__ == "__main__":
    main()
