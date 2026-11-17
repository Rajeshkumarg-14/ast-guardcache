"""
Validate the benchmark workload before any model run.

    uv run python -X utf8 scripts/validate_workload.py > results/workload_validation.txt

For every question: the gold query executes on the benchmark database, it is
tenant filtered (same rows on the full database and on the caller's tenant-only
copy), derived questions point at an earlier question of the same tenant, and a
gold query with ORDER BY ... LIMIT n has no tie at the cut-off. Exits non-zero on
any failure.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlglot
from sqlglot import exp

from app.database import execute_sql, reset_connection, tenant_only_connection
from scripts.workload import BENCHMARK_QUERIES, CATEGORIES, seed_sql


from app.database import canon_rows as canon  # noqa: E402


def tie_at_cutoff(sql: str) -> bool:
    tree = sqlglot.parse_one(sql, read="sqlite")
    limit = tree.args.get("limit")
    order = tree.args.get("order")
    if not limit or not order:
        return False
    n = int(limit.expression.name)
    keys = tree.copy()
    keys.set("expressions", [o.this.copy() for o in order.expressions])
    keys.set("limit", exp.Limit(expression=exp.Literal.number(n + 1)))
    vals = [tuple(r.values()) for r in execute_sql(keys.sql(dialect="sqlite"))]
    return len(vals) > n and vals[n - 1] == vals[n]


def main() -> int:
    reset_connection(seed_sql())
    iso = {t: tenant_only_connection(t) for t in (1, 2)}
    problems, empty = [], []
    print(f"{len(BENCHMARK_QUERIES)} questions")
    print("by category:", dict(Counter(q["category"] for q in BENCHMARK_QUERIES)))
    print("by tenant:  ", dict(Counter(q["tenant_id"] for q in BENCHMARK_QUERIES)))
    assert set(q["category"] for q in BENCHMARK_QUERIES) <= set(CATEGORIES)
    print()
    for q in BENCHMARK_QUERIES:
        try:
            rows = execute_sql(q["gold_sql"])
            scoped = canon(execute_sql(q["gold_sql"], conn=iso[q["tenant_id"]])) == canon(rows)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"Q{q['id']}: gold does not execute: {exc}")
            continue
        if not scoped:
            problems.append(f"Q{q['id']}: gold is not tenant filtered")
        if tie_at_cutoff(q["gold_sql"]):
            problems.append(f"Q{q['id']}: tie at the LIMIT cut-off")
        src = q["source"]
        if src is not None:
            s = BENCHMARK_QUERIES[src - 1]
            if src >= q["id"] or s["tenant_id"] != q["tenant_id"]:
                problems.append(f"Q{q['id']}: bad source Q{src}")
        n = len(rows)
        if n == 0 or all(v in (None, 0) for r in rows for v in r.values()):
            empty.append(q["id"])
        derived = f"  <- Q{src}" if src else ""
        print(f"Q{q['id']:>3} t{q['tenant_id']} {q['category']:<14} rows={n:<3} {q['question']}{derived}")
    print()
    print(f"gold queries with an empty or all-zero/NULL result: {len(empty)} {empty}")
    print(f"problems: {len(problems)}")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
