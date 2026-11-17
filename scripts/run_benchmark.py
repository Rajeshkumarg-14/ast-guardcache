"""
Automated 5-Mode Ablation Benchmark.

Runs the workload in scripts/workload.py across the pipeline ablation modes and produces:
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
import json
from collections import Counter
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

from app.database import canon_rows, execute_sql, norm_value, reset_connection, tenant_only_connection
from app.llm import API_ERROR_PREFIX
from app.pipeline import Pipeline

from scripts.workload import BENCHMARK_QUERIES, seed_sql  # noqa: E402  (re-exported)

MODES = ["zero_shot", "hybrid_rag", "rag_normalize", "rag_ast", "ast_guardcache"]
MODE_LABELS = {
    "zero_shot": "Zero-Shot",
    "hybrid_rag": "Hybrid RAG",
    "rag_normalize": "RAG + Re-print",
    "rag_ast": "RAG + AST",
    "ast_guardcache": "AST-GuardCache",
}


_canon = canon_rows


def check_tenant_leak(sql: str, iso_conn) -> bool:
    """
    Differential leak test: the executed SQL leaks iff its result on the full
    database differs from its result on a copy holding only the caller's rows.
    Catches aggregates over other tenants, not just projected tenant_id columns.
    """
    # No try/except: the SQL already ran on the full database, so a failure on the
    # copy would be a fault of the oracle and must not be read as "no leak".
    return _canon(execute_sql(sql)) != _canon(execute_sql(sql, conn=iso_conn))


def check_execution_accuracy(rows: list[dict], gold_sql: str) -> bool:
    """
    Execution accuracy against the tenant-filtered gold query.

    Correct iff predicted and gold results have the same number of rows and
    gold rows can be matched one-to-one to predicted rows that contain all of
    the gold row's values (as a multiset). Extra projected columns are
    tolerated; missing, extra or duplicated rows are not.
    """
    gold_rows = execute_sql(gold_sql)
    if len(rows) != len(gold_rows):
        return False
    pred = [Counter(str(norm_value(v)) for v in r.values()) for r in rows]
    gold = [Counter(str(norm_value(v)) for v in g.values()) for g in gold_rows]
    # Maximum bipartite matching (augmenting paths): a greedy first-fit can miss a
    # valid pairing when one predicted row could serve several gold rows.
    fits = [[j for j, p in enumerate(pred) if not need - p] for need in gold]
    owner: dict[int, int] = {}  # predicted row -> gold row

    def place(i: int, seen: set[int]) -> bool:
        for j in fits[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or place(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return all(place(i, set()) for i in range(len(gold)))


def executed(sql: str | None, error: str | None) -> bool:
    """A request counts as executed only if the model produced SQL and it ran."""
    return error is None and bool((sql or "").strip())


def run_benchmark(use_mock: bool = True, log_path: str | None = None, modes: list[str] | None = None) -> None:
    """Execute the full benchmark suite."""
    global MODES
    MODES = modes or MODES
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Fresh database with the benchmark-only seed rows
    reset_connection(seed_sql())

    pipeline = Pipeline(use_mock_llm=use_mock)
    iso = {t: tenant_only_connection(t) for t in (1, 2)}

    # Collect results per mode
    all_results: dict[str, list[dict]] = {mode: [] for mode in MODES}

    print("\n" + "=" * 80)
    print("  AST-GuardCache — 5-Mode Ablation Benchmark")
    print(f"  LLM Mode: {'Mock (Deterministic)' if use_mock else 'Live (OpenRouter)'}")
    print("=" * 80 + "\n")

    if not use_mock:
        # Untimed warm-up: the first request otherwise pays for loading the model
        pipeline.run(question="Count total active students", tenant_id=1, mode="zero_shot")

    # Interleaved: each question runs in every mode before the next question,
    # with the mode order rotated per question, so that drift in the endpoint
    # (load, thermal state) does not fall on one mode. Each mode still sees the
    # questions in workload order, which is what the cache depends on.
    pipeline.cache.clear()
    for i, q in enumerate(BENCHMARK_QUERIES):
        for mode in MODES[i % len(MODES):] + MODES[:i % len(MODES)]:
            result = pipeline.run(
                question=q["question"],
                tenant_id=q["tenant_id"],
                mode=mode,
            )
            if (result.error or "").startswith(API_ERROR_PREFIX):
                # Never score a provider outage as a model failure
                sys.exit(f"Aborting, results would be invalid: {result.error}")
            ran = executed(result.verified_sql, result.error)  # empty output is not an answer
            leaked = ran and check_tenant_leak(result.verified_sql, iso[q["tenant_id"]])
            accurate = ran and not leaked and check_execution_accuracy(
                result.execution_rows, q["gold_sql"]
            )

            all_results[mode].append(
                {
                    "query_idx": i + 1,
                    "question": q["question"],
                    "tenant_id": q["tenant_id"],
                    "category": q["category"],
                    "source": q["source"],
                    "accurate": accurate,
                    "leaked": leaked,
                    "cache_hit": result.cache_hit,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                    "pruned_tables": result.pruned_tables_count,
                    "retrieved_tables": result.retrieved_tables,
                    "verify_ms": round(result.verify_ms, 3),
                    "prompt_tokens": result.prompt_tokens,
                    "finish_reason": result.finish_reason,
                    "retries": result.ast_retries,
                    "generated_sql": result.generated_sql,
                    "executed_sql": result.verified_sql,
                }
            )
            status = "OK" if accurate and not leaked else "FAIL"
            print(f"  [{status}] Q{i+1} {MODE_LABELS[mode]}: {q['question'][:45]}  "
                  f"({result.latency_ms:.1f}ms, cache={'HIT' if result.cache_hit else 'MISS'})", flush=True)

    # ---------------------------------------------------------------------------
    # Aggregate metrics per mode
    # ---------------------------------------------------------------------------
    if log_path:
        _write_run_metadata(log_path, use_mock, started)
        with open(log_path, "w", encoding="utf-8") as f:
            for mode in MODES:
                for e in all_results[mode]:
                    f.write(json.dumps({"mode": mode, **e}) + "\n")

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
        hit_lats = [e["latency_ms"] for e in entries if e["cache_hit"]]
        miss_tables = [e["pruned_tables"] for e in entries if not e["cache_hit"]]

        summary.append(
            {
                "Mode": MODE_LABELS[mode],
                "Exec Accuracy %": f"{exec_acc:.1f}",
                "Tenant Leak Rate %": f"{leak_rate:.1f}",
                "Cache Hit Ratio %": f"{cache_hits:.1f}",
                "Avg Latency ms": f"{avg_lat:.1f}",
                # robust to the first request of a run, which may include model loading
                "Median Latency ms": f"{np.median(latencies):.1f}",
                "P95 Latency ms": f"{p95_lat:.1f}",
                "Cache Hit Latency ms": f"{np.mean(hit_lats):.1f}" if hit_lats else "-",
                "Avg DDL Tables": f"{np.mean(miss_tables):.1f}",
                "Avg AST Retries": f"{np.mean([e['retries'] for e in entries]):.2f}",
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


def _write_run_metadata(log_path: str, use_mock: bool, started: str) -> None:
    """Record what produced a log: model, endpoint, library versions, time."""
    import hashlib
    import platform
    from datetime import datetime, timezone
    from importlib.metadata import version

    import httpx

    from app.config import settings

    meta = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # fingerprint of the questions, gold SQL and categories this run used
        "workload_sha256": hashlib.sha256(
            json.dumps(BENCHMARK_QUERIES, sort_keys=True, default=str).encode()).hexdigest(),
        "model": "mock" if use_mock else settings.OPENROUTER_MODEL,
        "endpoint": None if use_mock else settings.OPENROUTER_BASE_URL,
        "temperature": 0.0,
        "max_output_tokens": 2048,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("sqlglot", "sentence-transformers", "rank-bm25", "openai", "torch")},
        "embedding_model": settings.EMBEDDING_MODEL,
        "wide_schema": os.getenv("WIDE_SCHEMA") == "1",
        "modes": MODES,
        "questions": len(BENCHMARK_QUERIES),
        "embedding_device": "cpu",
    }
    try:
        import torch

        meta["embedding_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    if not use_mock and ":11434" in settings.OPENROUTER_BASE_URL:  # Ollama: add server and model details
        base = settings.OPENROUTER_BASE_URL.rsplit("/v1", 1)[0]
        try:
            meta["ollama_version"] = httpx.get(f"{base}/api/version", timeout=5).json().get("version")
            show = httpx.post(f"{base}/api/show", json={"model": settings.OPENROUTER_MODEL}, timeout=10).json()
            meta["model_details"] = show.get("details")
            tags = httpx.get(f"{base}/api/tags", timeout=5).json().get("models", [])
            meta["model_digest"] = next((m["digest"] for m in tags if m["name"] == settings.OPENROUTER_MODEL), None)
        except Exception as exc:  # metadata is best effort
            meta["ollama_error"] = str(exc)
    with open(os.path.splitext(log_path)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _generate_chart(summary: list[dict], output_path: str) -> None:
    """Generate a grouped bar chart suitable for an IEEE paper."""
    # IEEE column width is ~3.5in, double column ~7.16in
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    modes = [s["Mode"] for s in summary]
    x = np.arange(len(modes))
    width = 0.55

    # Color palette
    colors = ["#94a3b8", "#60a5fa", "#f59e0b", "#818cf8", "#34d399"][: len(modes)]

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
    p95_colors = ["#b0b8c8", "#93bbf0", "#fbbf24", "#a5a8f0", "#7ee8c0"][: len(modes)]
    ax3.bar(x - bar_w / 2, avg_vals, bar_w, label="Avg Latency", color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax3.bar(x + bar_w / 2, p95_vals, bar_w, label="P95 Latency", color=p95_colors, edgecolor="white", linewidth=0.5, alpha=0.6)
    ax3.set_ylabel("Latency (ms)", fontsize=9)
    ax3.set_title("Query Latency Distribution", fontsize=10, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(modes, rotation=20, ha="right", fontsize=7)
    ax3.legend(fontsize=7)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"AST-GuardCache: Ablation Benchmark ({len(BENCHMARK_QUERIES)}-Query Suite)",
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
    parser.add_argument("--log", help="write per-query results (incl. SQL) as JSON lines")
    parser.add_argument("--modes", help="comma-separated subset of modes (default: all five)")
    args = parser.parse_args()
    run_benchmark(use_mock=args.mock_llm, log_path=args.log,
                  modes=args.modes.split(",") if args.modes else None)
