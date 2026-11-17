"""
Baseline A (outer WHERE) versus the current verifier on the generated SQLite
queries of tests/test_scenarios.py, with the same differential oracle.

    uv run python -X utf8 scripts/compare_rewriters.py --out results/rewriter_comparison

For every (query, tenant) case the oracle is the original query on a copy of
the database holding only that tenant's rows. Cases whose oracle does not run
(the generator built SQL the engine refuses) are excluded, as in the tests.
Per rewriter a case is: rejected, engine error (the scoped query fails while
the oracle runs), isolated (same rows) or leaked (different rows). Also writes
the verifier's own time per query.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import execute_sql
from app.verifier import verify_and_inject_ast
from scripts.baseline_rewriter import SET_OP_REJECTION, baseline_scope, features
from tests.test_scenarios import (
    COLUMNS, FKS, FUZZ, JOIN_SHAPES, KEY, PAIRS, SHAPES, TABLES, TENANTS, canon, iso,
)


def generated_cases() -> list[tuple[str, int, str]]:
    """(sql, tenant, family) exactly as the scenario tests build them."""
    out = []
    for t in TENANTS:
        other = 2 if t == 1 else 1
        out += [(s.format(t=tb, k=KEY[tb], me=t, other=other), t, "shapes") for s in SHAPES for tb in TABLES]
        out += [(s.format(c=c, cc=cc, p=p, pc=pc), t, "fk joins") for s in JOIN_SHAPES for c, cc, p, pc in FKS]
        for a, b in PAIRS:
            out.append((f"SELECT COUNT(*) FROM {a} AS a, {b} AS b", t, "table pairs"))
            out.append((f"SELECT COUNT(*) FROM {a} AS a WHERE EXISTS (SELECT 1 FROM {b} AS b WHERE b.id >= a.id)",
                        t, "table pairs"))
        for tb, col in COLUMNS:
            out.append((f"SELECT {col} FROM {tb}", t, "columns"))
            out.append((f"SELECT x.{col} FROM {tb} AS x ORDER BY x.{col}", t, "columns"))
        out += [(q, t, "random nested") for q in FUZZ]
    return out


def rows(sql: str, conn=None):
    try:
        return canon(execute_sql(sql, conn=conn))
    except sqlite3.Error as exc:
        return f"error: {exc}"


def outcome(result, expected) -> tuple[str, str | None]:
    if not result.valid:
        return "rejected", result.error
    got = rows(result.sql)
    if isinstance(got, str):
        return "engine error", got
    return ("isolated" if got == expected else "leaked"), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rewriter_comparison")
    args = ap.parse_args()

    cases = generated_cases()
    stats = {"ours": Counter(), "baseline": Counter()}
    by_feature = {"ours": Counter(), "baseline": Counter()}
    feature_total = Counter()
    setop_rejections = 0
    timings = {"ours": [], "baseline": []}
    distinct = set()
    excluded = 0
    log = open(args.out + ".jsonl", "w", encoding="utf-8")
    for sql, t, family in cases:
        expected = rows(sql, conn=iso(t))
        if isinstance(expected, str):
            excluded += 1
            continue
        feats = features(sql)
        feature_total.update(feats)
        distinct.add((sql, t))
        rec = {"sql": sql, "tenant": t, "family": family, "features": feats}
        for name, fn in (("ours", verify_and_inject_ast), ("baseline", baseline_scope)):
            t0 = time.perf_counter()
            result = fn(sql, t)
            timings[name].append((time.perf_counter() - t0) * 1000)
            o, err = outcome(result, expected)
            stats[name][o] += 1
            if o != "isolated":
                by_feature[name].update(feats)
            if name == "baseline" and err == SET_OP_REJECTION:
                setop_rejections += 1
            rec[name] = {"outcome": o, "error": err, "scoped": result.sql if result.valid else None}
        log.write(json.dumps(rec) + "\n")
    log.close()

    n = sum(stats["ours"].values())
    summary = {
        "cases_built": len(cases), "excluded_not_executable": excluded, "compared": n,
        "distinct_compared": len(distinct),
        "outcomes": {k: dict(v) for k, v in stats.items()},
        "baseline_setop_rejections": setop_rejections,
        "feature_total": dict(feature_total),
        "not_isolated_by_feature": {k: dict(v) for k, v in by_feature.items()},
        "time_ms": {k: {"median": statistics.median(v), "p95": sorted(v)[int(0.95 * len(v))], "max": max(v),
                        "n": len(v)} for k, v in timings.items()},
    }
    json.dump(summary, open(args.out + ".json", "w", encoding="utf-8"), indent=1)
    lines = [f"{len(cases)} generated (query, tenant) cases; {excluded} excluded (oracle not executable); "
             f"{n} compared ({len(distinct)} distinct)"]
    for k in ("isolated", "leaked", "engine error", "rejected"):
        lines.append(f"  {k:<13} ours {stats['ours'][k]:>5}   baseline {stats['baseline'][k]:>5}")
    lines.append(f"baseline rejections caused by a set operation: {setop_rejections}")
    lines.append("cases not isolated, by query feature (a case can have several):")
    for f, total in feature_total.most_common():
        lines.append(f"  {f:<17} {total:>5} cases   ours {by_feature['ours'][f]:>4}   baseline {by_feature['baseline'][f]:>4}")
    for k, v in summary["time_ms"].items():
        lines.append(f"rewrite time per query ({k}): median {v['median']:.2f} ms, p95 {v['p95']:.2f} ms, "
                     f"max {v['max']:.2f} ms, n={v['n']}")
    open(args.out + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
