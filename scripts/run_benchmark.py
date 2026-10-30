"""
Automated 4-Mode Ablation Benchmark.

Runs a 12-query benchmark suite across all 4 pipeline ablation modes and produces:
1. Terminal Markdown comparison table
2. benchmark_results.csv
3. ieee_benchmark_chart.pdf (grouped bar chart for the IEEE paper)

Usage:
    uv run python scripts/run_benchmark.py            # Live LLM
    uv run python scripts/run_benchmark.py --mock-llm  # Deterministic offline
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from app.database import get_connection, reset_connection
from app.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 12-Query Benchmark Suite
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: list[dict] = [
    # -- Basic retrieval queries --
    {
        "question": "Show all students in CSE with cgpa > 8.5",
        "tenant_id": 1,
        "ground_truth_contains": ["Alice Johnson", "Bob Smith"],
        "ground_truth_excludes_tenant": 2,
    },
    {
        "question": "List students in ECE with cgpa > 7.0",
        "tenant_id": 1,
        "ground_truth_contains": ["Charlie Lee"],
        "ground_truth_excludes_tenant": 2,
    },
    {
        "question": "Show students in CSE with cgpa > 9.0",
        "tenant_id": 2,
        "ground_truth_contains": ["Fatima Khan"],
        "ground_truth_excludes_tenant": 1,
    },
    # -- Aggregation queries --
    {
        "question": "Count total active students",
        "tenant_id": 1,
        "ground_truth_contains": [],
        "ground_truth_excludes_tenant": 2,
    },
    {
        "question": "What is the average cgpa of all students?",
        "tenant_id": 1,
        "ground_truth_contains": [],
        "ground_truth_excludes_tenant": 2,
    },
    # -- JOIN queries --
    {
        "question": "Show assessment scores with student names and tracks",
        "tenant_id": 1,
        "ground_truth_contains": [],
        "ground_truth_excludes_tenant": 2,
    },
    {
        "question": "List hackathon submissions with project titles and scores",
        "tenant_id": 1,
        "ground_truth_contains": [],
        "ground_truth_excludes_tenant": 2,
    },
    # -- FinTech / Billing queries --
    {
        "question": "Show pending billing invoices",
        "tenant_id": 1,
        "ground_truth_contains": [],
        "ground_truth_excludes_tenant": 2,
    },
    {
        "question": "Show overdue billing invoices",
        "tenant_id": 2,
        "ground_truth_contains": [],
        "ground_truth_excludes_tenant": 1,
    },
    # -- Operational queries --
    {
        "question": "Show failed webhook events",
        "tenant_id": 2,
        "ground_truth_contains": [],
        "ground_truth_excludes_tenant": 1,
    },
    {
        "question": "List queued notifications",
        "tenant_id": 1,
        "ground_truth_contains": [],
        "ground_truth_excludes_tenant": 2,
    },
    # -- Semantic cache variation (same intent, different params) --
    {
        "question": "Show students in ME with cgpa > 7.5",
        "tenant_id": 1,
        "ground_truth_contains": ["Diana Patel"],
        "ground_truth_excludes_tenant": 2,
    },
]

MODES = ["zero_shot", "hybrid_rag", "rag_ast", "ast_guardcache"]
MODE_LABELS = {
    "zero_shot": "Zero-Shot",
    "hybrid_rag": "Hybrid RAG",
    "rag_ast": "RAG + AST",
    "ast_guardcache": "AST-GuardCache",
}


def check_tenant_leak(rows: list[dict], excluded_tenant: int) -> bool:
    """Return True if any row contains data from the excluded tenant."""
    for row in rows:
        if "tenant_id" in row and row["tenant_id"] == excluded_tenant:
            return True
    return False


def check_execution_accuracy(rows: list[dict], expected_names: list[str]) -> bool:
    """
    Return True if all expected names appear in the result rows.
    If expected_names is empty, we just check the query didn't error.
    """
    if not expected_names:
        return True  # No specific assertion, just check it ran

    row_values = set()
    for row in rows:
        for v in row.values():
            if isinstance(v, str):
                row_values.add(v)

    return all(name in row_values for name in expected_names)


def run_benchmark(use_mock: bool = True) -> None:
    """Execute the full benchmark suite."""
    # Reset DB for clean state
    reset_connection()

    pipeline = Pipeline(use_mock_llm=use_mock)

    # Collect results per mode
    all_results: dict[str, list[dict]] = {mode: [] for mode in MODES}

    print("\n" + "=" * 80)
    print("  AST-GuardCache — 4-Mode Ablation Benchmark")
    print(f"  LLM Mode: {'Mock (Deterministic)' if use_mock else 'Live (OpenRouter)'}")
    print("=" * 80 + "\n")

    for mode in MODES:
        print(f"\n> Running mode: {MODE_LABELS[mode]}")
        # Reset cache between modes so cache hits are mode-specific
        pipeline.cache.clear()

        for i, q in enumerate(BENCHMARK_QUERIES):
            result = pipeline.run(
                question=q["question"],
                tenant_id=q["tenant_id"],
                mode=mode,
            )
            leaked = check_tenant_leak(
                result.execution_rows, q["ground_truth_excludes_tenant"]
            )
            accurate = check_execution_accuracy(
                result.execution_rows, q["ground_truth_contains"]
            ) and result.error is None

            all_results[mode].append(
                {
                    "query_idx": i + 1,
                    "question": q["question"][:50],
                    "tenant_id": q["tenant_id"],
                    "accurate": accurate,
                    "leaked": leaked,
                    "cache_hit": result.cache_hit,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                    "pruned_tables": result.pruned_tables_count,
                }
            )
            status = "OK" if accurate and not leaked else "FAIL"
            print(f"  [{status}] Q{i+1}: {q['question'][:45]}  "
                  f"({result.latency_ms:.1f}ms, cache={'HIT' if result.cache_hit else 'MISS'})")

    # ---------------------------------------------------------------------------
    # Aggregate metrics per mode
    # ---------------------------------------------------------------------------
    summary: list[dict] = []
    for mode in MODES:
        entries = all_results[mode]
        n = len(entries)
        exec_acc = sum(1 for e in entries if e["accurate"]) / n * 100
        leak_rate = sum(1 for e in entries if e["leaked"]) / n * 100
        cache_hits = sum(1 for e in entries if e["cache_hit"]) / n * 100
        latencies = [e["latency_ms"] for e in entries]
        avg_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)

        summary.append(
            {
                "Mode": MODE_LABELS[mode],
                "Exec Accuracy %": f"{exec_acc:.1f}",
                "Tenant Leak Rate %": f"{leak_rate:.1f}",
                "Cache Hit Ratio %": f"{cache_hits:.1f}",
                "Avg Latency ms": f"{avg_lat:.1f}",
                "P95 Latency ms": f"{p95_lat:.1f}",
            }
        )

    # ---------------------------------------------------------------------------
    # Terminal output
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)
    print(tabulate(summary, headers="keys", tablefmt="github"))

    # ---------------------------------------------------------------------------
    # CSV export
    # ---------------------------------------------------------------------------
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark_results.csv",
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n[OK] CSV saved to: {csv_path}")

    # ---------------------------------------------------------------------------
    # IEEE-quality PDF chart
    # ---------------------------------------------------------------------------
    pdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ieee_benchmark_chart.pdf",
    )
    _generate_chart(summary, pdf_path)
    print(f"[OK] PDF chart saved to: {pdf_path}\n")


def _generate_chart(summary: list[dict], output_path: str) -> None:
    """Generate a grouped bar chart suitable for an IEEE paper."""
    # IEEE column width is ~3.5in, double column ~7.16in
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    modes = [s["Mode"] for s in summary]
    x = np.arange(len(modes))
    width = 0.55

    # Color palette
    colors = ["#94a3b8", "#60a5fa", "#818cf8", "#34d399"]

    # Chart 1: Exec Accuracy vs Tenant Leak
    ax1 = axes[0]
    acc_vals = [float(s["Exec Accuracy %"]) for s in summary]
    leak_vals = [float(s["Tenant Leak Rate %"]) for s in summary]
    bar_w = 0.35
    ax1.bar(x - bar_w / 2, acc_vals, bar_w, label="Exec Accuracy %", color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax1.bar(x + bar_w / 2, leak_vals, bar_w, label="Tenant Leak %", color="#f87171", edgecolor="white", linewidth=0.5, alpha=0.7)
    ax1.set_ylabel("Percentage (%)", fontsize=9)
    ax1.set_title("Execution Accuracy & Tenant Leak Rate", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, rotation=20, ha="right", fontsize=7)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", alpha=0.3)

    # Chart 2: Cache Hit Ratio
    ax2 = axes[1]
    cache_vals = [float(s["Cache Hit Ratio %"]) for s in summary]
    bars = ax2.bar(x, cache_vals, width, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, cache_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Cache Hit Ratio (%)", fontsize=9)
    ax2.set_title("Parameterized Semantic Cache Efficacy", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(modes, rotation=20, ha="right", fontsize=7)
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", alpha=0.3)

    # Chart 3: Latency
    ax3 = axes[2]
    avg_vals = [float(s["Avg Latency ms"]) for s in summary]
    p95_vals = [float(s["P95 Latency ms"]) for s in summary]
    p95_colors = ["#b0b8c8", "#93bbf0", "#a5a8f0", "#7ee8c0"]
    ax3.bar(x - bar_w / 2, avg_vals, bar_w, label="Avg Latency", color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax3.bar(x + bar_w / 2, p95_vals, bar_w, label="P95 Latency", color=p95_colors, edgecolor="white", linewidth=0.5, alpha=0.6)
    ax3.set_ylabel("Latency (ms)", fontsize=9)
    ax3.set_title("Query Latency Distribution", fontsize=10, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(modes, rotation=20, ha="right", fontsize=7)
    ax3.legend(fontsize=7)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "AST-GuardCache: 4-Mode Ablation Benchmark (12-Query Suite)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AST-GuardCache Ablation Benchmark"
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic mock LLM instead of calling OpenRouter",
    )
    args = parser.parse_args()
    run_benchmark(use_mock=args.mock_llm)
