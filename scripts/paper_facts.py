"""
Regenerate the one-off numbers quoted in the paper that no table reports.

    uv run python scripts/paper_facts.py results/*.jsonl [--bird <bird>/dev_20240627]

- cosine similarities used to motivate the cache design
- whether the retriever returned every table of the gold query (from the logs)
- time spent in the verifier per request (from the logs)
- structure of the BIRD gold queries (joins, subqueries, CTEs, ...)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sqlglot
from sqlglot import exp

from app.cache import strip_entities
from app.retriever import _get_model
from scripts.run_benchmark import BENCHMARK_QUERIES


def cosine(a: str, b: str) -> float:
    e = _get_model().encode([a, b], normalize_embeddings=True)
    return float(e[0] @ e[1])


def similarities() -> None:
    print("cosine similarity, all-MiniLM-L6-v2")
    a, b = "List CSE students with CGPA > 8.5", "List ECE students with CGPA > 9.0"
    print(f"  raw questions, different constants:   {cosine(a, b):.3f}")
    for a, b in [("Show assessments with score > 80", "Show assessments with score < 80"),
                 ("Show the top 5 students by cgpa", "Show the bottom 5 students by cgpa")]:
        ma, mb = strip_entities(a)[0], strip_entities(b)[0]
        print(f"  masked: {ma!r} vs {mb!r}: {cosine(ma, mb):.3f}")
    print("  derived questions whose normalised text equals that of their source, by category:")
    same: dict[str, list[int]] = {}
    for q in BENCHMARK_QUERIES:
        if q["source"]:
            src = BENCHMARK_QUERIES[q["source"] - 1]["question"]
            ok = strip_entities(q["question"])[0] == strip_entities(src)[0]
            same.setdefault(q["category"], [0, 0])
            same[q["category"]][0] += ok
            same[q["category"]][1] += 1
    for cat, (k, n) in same.items():
        print(f"    {cat}: {k} of {n}")


def from_logs(paths: list[str]) -> None:
    gold_tables = [
        {t.name for t in sqlglot.parse_one(q["gold_sql"], read="sqlite").find_all(exp.Table)}
        for q in BENCHMARK_QUERIES
    ]
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        retrieved = [r for r in rows if r["mode"] != "zero_shot" and not r["cache_hit"] and "retrieved_tables" in r]
        missed = sum(not gold_tables[r["query_idx"] - 1] <= set(r["retrieved_tables"]) for r in retrieved)
        verify = [r["verify_ms"] for r in rows if r.get("verify_ms")]
        sizes = [len(r["retrieved_tables"]) for r in retrieved]
        print(f"{os.path.basename(path)}:")
        if retrieved:
            print(f"  retrieval: {len(retrieved)} requests, gold tables missing in {missed}; "
                  f"tables per prompt mean {np.mean(sizes):.1f} (min {min(sizes)}, max {max(sizes)})")
        if verify:
            print(f"  verifier time per request: median {np.median(verify):.2f} ms, "
                  f"95th percentile {np.percentile(verify, 95):.2f} ms, max {max(verify):.2f} ms (n={len(verify)})")
        hits = [r["latency_ms"] for r in rows if r["cache_hit"]]
        if hits:
            print(f"  cache hits: n={len(hits)}, mean {np.mean(hits):.1f} ms, range {min(hits):.1f}-{max(hits):.1f} ms")
        empty = [(r["mode"], r["query_idx"], r.get("finish_reason")) for r in rows
                 if not r["generated_sql"] and not r["cache_hit"]]
        if empty:
            print(f"  empty model output: {empty}")


def bird_structure(bird_dir: str) -> None:
    items = json.load(open(os.path.join(bird_dir, "dev.json"), encoding="utf-8"))
    count = lambda pat: sum(bool(re.search(pat, x["SQL"], re.I)) for x in items)  # noqa: E731
    print(f"BIRD dev gold queries: {len(items)}")
    for label, pat in [("with a JOIN", r"\bJOIN\b"), ("with an outer join", r"\b(LEFT|RIGHT|FULL)\s+(OUTER\s+)?JOIN\b"),
                       ("with a nested SELECT", r"\(\s*SELECT\b"), ("starting with WITH (CTE)", r"^\s*WITH\b"),
                       ("with a window function", r"\bOVER\s*\("), ("with UNION/INTERSECT/EXCEPT", r"\b(UNION|INTERSECT|EXCEPT)\b")]:
        print(f"  {label}: {count(pat)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="benchmark JSON-lines logs")
    ap.add_argument("--bird", help="folder holding BIRD dev.json")
    args = ap.parse_args()
    similarities()
    from_logs(args.logs)
    if args.bird:
        bird_structure(args.bird)
