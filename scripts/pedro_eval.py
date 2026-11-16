"""
The published table-replacement rewriter of Pedro et al., scored with the same
differential oracle as our verifier, on the generated queries and on BIRD.

    uv run python -X utf8 scripts/pedro_eval.py generated --out results/pedro_generated
    uv run python -X utf8 scripts/pedro_eval.py bird --bird-dir <bird>/dev_20240627 --db-dir <bird_out> --out results/pedro_bird

Their file runs unchanged, under the sqlglot version their artifact pins, in a
worker process (scripts/third_party/pedro_worker.py). For a fair comparison the
query first passes our read-only and catalog checks, as for the outer-WHERE
baseline; only the scoping step differs. The rewriter receives one clause
"<key> = <tenant>" per tenant table, under every spelling of the table name that
the query uses (it matches names case-sensitively).

Outcomes per case: rejected (our guard refused it), rewriter error (their code
raised), engine error (scoped query fails while the oracle runs; for BIRD also a
time-out on either side), isolated, leaked.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

from app.database import TENANT_SCOPED_TABLES  # noqa: E402
from app.verifier import verify_and_inject_ast  # noqa: E402
from scripts.baseline_rewriter import features  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "pedro_worker.py")


class Pedro:
    def __init__(self) -> None:
        self.p = subprocess.Popen(
            ["uv", "run", "--no-project", "--with", "sqlglot==20.10.0", "python", "-I", WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        )

    def __call__(self, sql: str, keys: dict[str, str], tenant: int) -> dict:
        """keys: lower-case table name -> tenant key column."""
        spellings = {t.name for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}
        names = {n for n in spellings if n.lower() in keys} | set(keys)
        clauses = [{"table_name": n, "where_clause": f"{keys[n.lower()]} = {int(tenant)}"} for n in sorted(names)]
        self.p.stdin.write(json.dumps({"sql": sql, "clauses": clauses}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())


def score(pedro: Pedro, sql: str, t: int, keys: dict, expected, run_full, schema=None) -> tuple[str, dict]:
    guard = verify_and_inject_ast(sql, t, schema=schema)
    if not guard.valid:
        return "rejected", {"error": guard.error}
    r = pedro(sql, keys, t)
    if "error" in r:
        return "rewriter error", r
    r["scoped"] = r.pop("sql")
    got = run_full(r["scoped"])
    if isinstance(got, str) or isinstance(expected, str):
        return "engine error", r | {"error": got if isinstance(got, str) else f"oracle: {expected}"}
    return ("isolated" if got == expected else "leaked"), r


def report(out: str, stats: Counter, by_feature: Counter, feature_total: Counter, ms: list, header: str) -> None:
    n = sum(stats.values())
    lines = [header]
    for k in ("isolated", "leaked", "engine error", "rewriter error", "rejected"):
        lines.append(f"  {k:<15} {stats[k]:>5}  ({100 * stats[k] / n:.1f}%)")
    lines.append("cases not isolated, by query feature (a case can have several):")
    for f, total in feature_total.most_common():
        lines.append(f"  {f:<17} {total:>5} cases   pedro {by_feature[f]:>4}")
    if ms:
        lines.append(f"rewrite time per query (pedro, sqlglot 20.10.0): median {statistics.median(ms):.2f} ms, "
                     f"p95 {sorted(ms)[int(0.95 * len(ms))]:.2f} ms, n={len(ms)}")
    json.dump({"outcomes": dict(stats), "not_isolated_by_feature": dict(by_feature),
               "feature_total": dict(feature_total),
               "time_ms_median": statistics.median(ms) if ms else None},
              open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def generated(out: str) -> None:
    from scripts.compare_rewriters import generated_cases, rows
    from tests.test_scenarios import iso

    keys = {t: "tenant_id" for t in TENANT_SCOPED_TABLES} | {"tenants": "id"}
    pedro = Pedro()
    stats, by_feature, feature_total, ms = Counter(), Counter(), Counter(), []
    excluded = 0
    log = open(out + ".jsonl", "w", encoding="utf-8")
    for sql, t, family in generated_cases():
        expected = rows(sql, conn=iso(t))
        if isinstance(expected, str):
            excluded += 1
            continue
        feats = features(sql)
        feature_total.update(feats)
        o, r = score(pedro, sql, t, keys, expected, rows)
        stats[o] += 1
        if "ms" in r:
            ms.append(r["ms"])
        if o != "isolated":
            by_feature.update(feats)
        log.write(json.dumps({"sql": sql, "tenant": t, "family": family, "outcome": o, **r}) + "\n")
    log.close()
    report(out, stats, by_feature, feature_total, ms,
           f"generated queries: {sum(stats.values())} compared, {excluded} excluded (oracle not executable)")


def bird(bird_dir: str, db_dir: str, out: str) -> None:
    from scripts import bird_tenant_eval as B

    items = json.load(open(os.path.join(bird_dir, "dev.json"), encoding="utf-8"))
    pedro = Pedro()
    stats, by_feature, feature_total, ms = Counter(), Counter(), Counter(), []
    conns = {}
    log = open(out + ".jsonl", "w", encoding="utf-8")
    for i, item in enumerate(items):
        db = item["db_id"]
        if db not in conns:
            src = os.path.join(bird_dir, "dev_databases", db, f"{db}.sqlite")
            dst = os.path.join(db_dir, f"{db}_tenants.sqlite")
            B.build_tenant_db(src, dst)
            full = B.connect(dst, None)
            conns[db] = (full, B.schema_of(full), {t: B.connect(dst, t) for t in B.TENANTS})
        full, schema, iso = conns[db]
        keys = {tb: "tenant_id" for tb, cols in schema.items() if "tenant_id" in cols}
        gold = item["SQL"]
        feats = features(gold)
        for t in B.TENANTS:
            feature_total.update(feats)
            expected = B.run(iso[t], gold)
            o, r = score(pedro, gold, t, keys, expected, lambda s: B.run(full, s), schema=schema)
            stats[o] += 1
            if "ms" in r:
                ms.append(r["ms"])
            if o != "isolated":
                by_feature.update(feats)
            log.write(json.dumps({"question_id": item.get("question_id", i), "db_id": db, "tenant": t,
                                  "gold": gold, "outcome": o, **r}) + "\n")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(items)} {dict(stats)}", flush=True)
    log.close()
    report(out, stats, by_feature, feature_total, ms,
           f"BIRD: {len(items)} gold queries x {len(B.TENANTS)} tenants = {sum(stats.values())} cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["generated", "bird"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bird-dir")
    ap.add_argument("--db-dir")
    a = ap.parse_args()
    generated(a.out) if a.which == "generated" else bird(a.bird_dir, a.db_dir, a.out)
