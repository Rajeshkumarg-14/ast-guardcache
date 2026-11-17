"""
Replay logged LLM SQL (run_benchmark.py --log) on PostgreSQL under four defences:
none, RLS only, verifier only, verifier + RLS.

Each distinct (generated_sql, tenant) is transpiled SQLite -> Postgres and
compared with the same query on the tenant-only schema (differential oracle).
Requires docker-compose.research.yml.

    python scripts/replay_pg.py live/*.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import psycopg
import sqlglot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import pg  # noqa: E402
from app.database import canon_rows  # noqa: E402
from app.verifier import verify_and_inject_ast  # noqa: E402
from scripts.workload import seed_sql  # noqa: E402

CONFIGS = ["none", "rls_only", "verifier_only", "verifier_rls"]
RAW_MODES = {"zero_shot", "hybrid_rag", "rag_normalize"}


def _rows(sql: str, **kw):
    try:
        return canon_rows(pg.execute(sql, **kw))
    except psycopg.Error:
        return None


def main(paths: list[str]) -> None:
    pg.setup(extra_sql=seed_sql())  # same rows as the benchmark database
    cases = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                # Unverified modes only: their generated_sql is the model's raw output
                if rec.get("generated_sql") and rec["mode"] in RAW_MODES:
                    cases.add((rec["generated_sql"], rec["tenant_id"]))

    stats = {c: Counter() for c in CONFIGS}
    for raw, tenant in sorted(cases):
        try:
            sql = sqlglot.transpile(raw, read="sqlite", write="postgres")[0]
        except sqlglot.errors.ParseError:
            for c in CONFIGS:
                stats[c]["unparseable"] += 1
            continue
        oracle = _rows(sql, schema=f"iso_{tenant}")
        verified = verify_and_inject_ast(sql, tenant, dialect="postgres")
        runs = {
            "none": _rows(sql),
            "rls_only": _rows(sql, rls_tenant=tenant),
            "verifier_only": _rows(verified.sql) if verified.valid else "rejected",
            "verifier_rls": _rows(verified.sql, rls_tenant=tenant) if verified.valid else "rejected",
        }
        for c, rows in runs.items():
            if rows == "rejected":
                stats[c]["rejected"] += 1
            elif rows is None or oracle is None:
                stats[c]["db_error"] += 1  # the query or its oracle did not run: no verdict
            elif rows != oracle:
                stats[c]["leaked"] += 1
            else:
                stats[c]["isolated"] += 1

    n = len(cases)
    print(f"{n} distinct (SQL, tenant) cases from {len(paths)} logs\n")
    print(f"{'config':<15}" + "".join(f"{k:>12}" for k in ["isolated", "leaked", "rejected", "db_error", "unparseable"]))
    for c in CONFIGS:
        print(f"{c:<15}" + "".join(f"{stats[c][k]:>12}" for k in ["isolated", "leaked", "rejected", "db_error", "unparseable"]))


if __name__ == "__main__":
    main(sys.argv[1:])
