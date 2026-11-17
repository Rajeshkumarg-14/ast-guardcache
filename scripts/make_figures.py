"""
Paper figures, built from the per-query logs written by run_benchmark.py --log.

    uv run python scripts/make_figures.py results/<primary>_run*.jsonl --out docs/figures

fig1_architecture.png  pipeline diagram (no data)
fig2_ablation.png      accuracy / leak rate and median / P95 latency per ablation mode
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8})

MODES = ["zero_shot", "hybrid_rag", "rag_normalize", "rag_ast", "ast_guardcache"]
LABELS = ["Zero-shot", "Hybrid RAG", "RAG +\nre-print", "RAG +\nAST", "AST-\nGuardCache"]


def architecture(path: str) -> None:
    """Single-column (3.5 in) pipeline diagram."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=600)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, shade="white", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc=shade, ec="black", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=6.5, fontweight="bold" if bold else "normal", linespacing=1.2)

    def arrow(a, b, label="", dashed=False, dx=0.0, dy=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color="black",
                                     linestyle="--" if dashed else "-"))
        if label:
            ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, label, ha="center", va="center", fontsize=6)

    box(1, 86, 27, 12, "Question Q\n+ tenant t\n(from session)", "#eeeeee", True)
    box(36, 86, 27, 12, "Literal stripper\nQ → Q', P")
    box(71, 86, 28, 12, "Template cache\n(per tenant)", "#eeeeee", True)

    box(1, 56, 38, 13, "Rebind P into cached\ntemplate; literal\ngrounding check")
    box(57, 61, 42, 14, "Hybrid retriever:\nBM25 + dense, RRF,\none-hop FK closure")
    box(57, 40, 42, 11, "LLM (untrusted\nSQL generator)")
    box(1, 17, 98, 12, "AST verifier (every path)\nread-only guard · catalog check · per-table tenant scoping", "#dddddd", True)
    box(26, 1, 48, 8, "Read-only execution on live data", "#eeeeee")

    arrow((28.5, 92), (35.5, 92))
    arrow((63.5, 92), (70.5, 92))
    arrow((74, 85.5), (30, 69.6))
    ax.text(20, 78.5, "hit: cos ≥ τ, |P| equal,\nsame operator cues", ha="center", va="center", fontsize=6)
    arrow((88, 85.5), (88, 75.6), "miss", dx=6)
    arrow((70, 60.5), (70, 51.5), "pruned DDL", dx=13)
    arrow((66, 39.5), (66, 29.5), "SQL", dx=-5)
    arrow((80, 29.5), (80, 39.5), "error\n(≤ 2 retries)", dashed=True, dx=11)
    arrow((20, 55.5), (20, 29.5), "bound SQL", dx=9)
    arrow((50, 16.5), (50, 9.5), "tenant-scoped SQL", dx=14)

    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def ablation(log_paths: list[str], path: str) -> None:
    # Several runs of one model are pooled (equal sizes, so the mean over runs)
    rows = [json.loads(line) for p in log_paths for line in open(p, encoding="utf-8")]
    by_mode = {m: [r for r in rows if r["mode"] == m] for m in MODES}
    acc = [100 * np.mean([r["accurate"] for r in by_mode[m]]) for m in MODES]
    leak = [100 * np.mean([r["leaked"] for r in by_mode[m]]) for m in MODES]
    # Median, not mean: the first request of a run includes loading the model
    med = [np.median([r["latency_ms"] for r in by_mode[m]]) for m in MODES]
    p95 = [np.percentile([r["latency_ms"] for r in by_mode[m]], 95) for m in MODES]

    x = np.arange(len(MODES))
    w = 0.38
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 2.9), dpi=600)  # single column

    b1 = a1.bar(x - w / 2, acc, w, color="#555555", label="Tenant-safe exec. accuracy")
    b2 = a1.bar(x + w / 2, leak, w, color="white", edgecolor="black", hatch="////", label="Cross-tenant leak rate")
    a1.bar_label(b1, fmt="%.1f", fontsize=6, padding=1)
    a1.bar_label(b2, fmt="%.1f", fontsize=6, padding=1)
    a1.set_ylabel("Queries (%)")
    a1.set_ylim(0, 140)
    a1.set_yticks(range(0, 101, 20))
    a1.legend(fontsize=6.5, loc="upper left", ncol=2, frameon=False)

    a2.bar(x - w / 2, med, w, color="#555555", label="Median")
    a2.bar(x + w / 2, p95, w, color="white", edgecolor="black", hatch="....", label="95th percentile")
    a2.set_ylabel("Latency (ms)")
    a2.set_ylim(0, max(p95) * 1.3)
    a2.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)

    for a, tag in ((a1, "(a)"), (a2, "(b)")):
        a.set_xticks(x)
        a.set_xticklabels(LABELS, fontsize=6.5)
        a.grid(axis="y", lw=0.3, alpha=0.5)
        a.set_axisbelow(True)
        a.set_xlabel(tag)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="JSON-lines logs (runs) of the primary model")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    architecture(os.path.join(args.out, "fig1_architecture.png"))
    ablation(args.logs, os.path.join(args.out, "fig2_ablation.png"))
    print("figures written to", args.out)
