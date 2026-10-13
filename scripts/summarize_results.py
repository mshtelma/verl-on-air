#!/usr/bin/env python3
"""Build the results overview from committed OfficeQA and MuSiQue evidence.

Run with --check to detect stale summary/table values, or --plot to regenerate
the PNG/SVG with matplotlib. No remote services or training jobs are called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN GENERATED: heldout results (scripts/summarize_results.py) -->"
END = "<!-- END GENERATED: heldout results -->"


def evidence(path: str) -> tuple[dict, dict]:
    raw = (REPO / path).read_bytes()
    return json.loads(raw), {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}


def summarize() -> dict:
    officeqa, office_source = evidence("results/officeqa/2026-10-11-final-comparison.json")
    search, search_source = evidence("results/agentic-search/2026-09-heldout-test.json")
    heldout = officeqa["splits"]["heldout"]
    primary = heldout["metrics"]["pass_at_1"]
    rows = [{
        "usecase": "OfficeQA", "run": "step 100", "metric": "Supported pass@1",
        "questions": heldout["questions"], "samples_per_question": heldout["samples_per_question"],
        "baseline": primary["baseline"], "trained": primary["final"],
        "delta": primary["delta"], "delta_ci95": primary["delta_ci95"],
        "checkpoint_selection": "fixed before training", "source": office_source,
        "report": "results/officeqa/2026-10-11-final-report.md",
        "runtime": "PyTorch 2.13.0 / verl 0.10.0.dev0 / vLLM 0.30.0",
    }]
    tools = search["policies"]["tools"]
    for index, checkpoint in enumerate(tools["checkpoints"], start=1):
        paired = checkpoint["paired"]
        if not checkpoint["valid"] or checkpoint["n_scored"] != paired["n"]:
            raise ValueError("search summary requires a valid, complete held-out evaluation")
        rows.append({
            "usecase": "MuSiQue", "run": f"run {index}, step 20", "metric": "Exact match",
            "questions": paired["n"], "samples_per_question": 1,
            "baseline": tools["base"]["em"], "trained": checkpoint["em"],
            "delta": paired["delta"], "delta_ci95": paired["delta_ci95"],
            "checkpoint_selection": checkpoint["chosen"], "source": search_source,
            "report": "RESULTS.md#held-out-test-500-questions-scored-once",
            "runtime": "earlier stack; historical September 2026 measurements",
            "exact_mcnemar_p": paired["exact_mcnemar_p"],
        })
    return {"schema": "verl-on-air.results/v1", "actor": "Qwen3.5-35B-A3B",
            "comparison_unit": "paired questions within each experiment; metrics and policies differ",
            "confidence_interval": "95% paired question bootstrap; excludes training-seed variability",
            "experiments": rows}


def interval(row: dict) -> str:
    low, high = row["delta_ci95"]
    return f"[{low * 100:+.2f}, {high * 100:+.2f}]"


def table(summary: dict) -> str:
    lines = [BEGIN, "", "| Use case / checkpoint | Held-out questions × samples | Metric | Baseline | Trained | Gain | Paired 95% interval |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in summary["experiments"]:
        label = f"[{row['usecase']}, {row['run']}]({row['report']})"
        lines.append(f"| {label} | {row['questions']} × {row['samples_per_question']} | {row['metric']} | "
                     f"{row['baseline'] * 100:.2f}% | {row['trained'] * 100:.2f}% | "
                     f"{row['delta'] * 100:+.2f} pp | {interval(row)} pp |")
    return "\n".join([*lines, "", END])


def plot(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.hashsalt": "verl-on-air/results/v1"})
    rows = summary["experiments"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.34, wspace=0.35)
    fig.suptitle("Tool agents: measured held-out results", x=0.08, y=0.96, ha="left", fontsize=23, fontweight="bold", color="#15243c")
    fig.text(0.08, 0.895, "Qwen3.5-35B-A3B  ·  Compare each run with its own baseline", fontsize=12, color="#526075")
    panels = [
        ("OfficeQA", "Supported pass@1 (%)", [rows[0]["baseline"], rows[0]["trained"]], ["Baseline", "Step 100"], 30),
        ("MuSiQue retrieval", "Exact match (%)", [rows[1]["baseline"], rows[1]["trained"], rows[2]["trained"]], ["Baseline", "Run 1\nstep 20", "Run 2\nstep 20"], 50),
    ]
    for ax, (title, metric, scores, labels, ymax) in zip(axes, panels):
        values = [score * 100 for score in scores]
        colors = ["#a7b2c3", "#2761c8", "#153c87"][:len(values)]
        bars = ax.bar(range(len(values)), values, color=colors, width=0.58, zorder=3)
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=18, color="#15243c")
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, ymax)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6eaf0")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd2dc")
        ax.tick_params(length=0, pad=8)
        ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=7, fontsize=12, fontweight="bold")
    axes[0].text(0, -0.25, f"Gain {rows[0]['delta'] * 100:+.2f} pp  ·  95% CI {interval(rows[0])}\n43 hard questions × 4 samples  ·  one training seed",
                 transform=axes[0].transAxes, fontsize=10, color="#33445c", va="top")
    axes[1].text(0, -0.25, f"Run 1: {rows[1]['delta'] * 100:+.2f} pp  ·  95% CI {interval(rows[1])}\nRun 2: {rows[2]['delta'] * 100:+.2f} pp  ·  95% CI {interval(rows[2])}\n500 questions  ·  two training runs",
                 transform=axes[1].transAxes, fontsize=10, color="#33445c", va="top")
    fig.text(0.08, 0.04, "OfficeQA: fixed step 100. MuSiQue: run 1 chosen on dev; run 2 fixed before training.\n"
             "Separate metrics and policies. Intervals describe question uncertainty, not seed variability.\n"
             "OfficeQA AIR training receipt: FAILED; independently certified checkpoint. Evidence: RESULTS.md.",
             fontsize=9, color="#526075", linespacing=1.6)
    output = REPO / "results/heldout-summary"
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    svg = output.with_suffix(".svg")
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    summary = summarize()
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    results_path = REPO / "RESULTS.md"
    original = results_path.read_text()
    head, rest = original.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    updated = head + table(summary) + tail
    summary_path = REPO / "results/summary.json"
    if args.check:
        if not summary_path.is_file() or summary_path.read_text() != summary_text or original != updated:
            print("results overview is stale; run python3 scripts/summarize_results.py")
            return 1
        print("results ok  overview and table match committed held-out evidence")
        return 0
    summary_path.write_text(summary_text)
    results_path.write_text(updated)
    if args.plot:
        plot(summary)
    print("wrote results overview" + (" and PNG/SVG figure" if args.plot else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
