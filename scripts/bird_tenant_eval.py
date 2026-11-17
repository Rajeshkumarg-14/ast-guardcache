"""
Tenant-extended BIRD: run the verifier on human-written gold SQL from a real
benchmark and measure (a) false rejections and (b) isolation.

Each BIRD dev database is rebuilt with a ``tenant_id`` column on every table:
tenant 1 holds every original row, tenant 2 every second row, so an unscoped
query returns visibly different results. For each gold query Q and tenant t:

- rejected:  the verifier refuses Q (a false rejection: Q is valid SQL)
- isolated:  verify(Q, t) on the full database == Q on tenant t's rows only
- leaked:    the two differ
- error:     either side fails or times out

Each case is also scored for baseline A (scripts/baseline_rewriter.py, the
outer-WHERE scoping of our first version) against the same oracle.

"Tenant t's rows only" is a connection where a TEMP VIEW per table, filtered on
tenant_id = t, shadows the real table (SQLite resolves temp before main).

    python scripts/bird_tenant_eval.py <bird>/dev_20240627 --out bird_out
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import canon_rows  # noqa: E402
from app.verifier import verify_and_inject_ast  # noqa: E402
from scripts.baseline_rewriter import baseline_scope, features  # noqa: E402

TENANTS = (1, 2)
TIMEOUT_S = 20.0


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_tenant_db(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    conn = sqlite3.connect(tmp)
    conn.execute("ATTACH DATABASE ? AS src", (src,))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM src.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )]
    for t in tables:
        conn.execute(
            f"CREATE TABLE main.{_q(t)} AS "
            f"SELECT 1 AS tenant_id, * FROM src.{_q(t)} "
            f"UNION ALL SELECT 2 AS tenant_id, * FROM src.{_q(t)} WHERE rowid % 2 = 0"
        )
    conn.commit()
    conn.close()
    os.replace(tmp, dst)


def schema_of(conn: sqlite3.Connection) -> dict[str, list[str]]:
    out = {}
    for (t,) in conn.execute("SELECT name FROM main.sqlite_master WHERE type = 'table'"):
        out[t.lower()] = [r[1].lower() for r in conn.execute(f"PRAGMA main.table_info({_q(t)})")]
    return out


def connect(path: str, tenant: int | None) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    if tenant is not None:
        for (t,) in conn.execute("SELECT name FROM main.sqlite_master WHERE type = 'table'").fetchall():
            conn.execute(f"CREATE TEMP VIEW {_q(t)} AS SELECT * FROM main.{_q(t)} WHERE tenant_id = {tenant}")
    return conn


def run(conn: sqlite3.Connection, sql: str):
    deadline = time.monotonic() + TIMEOUT_S
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return canon_rows(conn.execute(sql).fetchall())
    except sqlite3.Error as exc:
        return f"error: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bird_dir", help="folder holding dev.json and dev_databases/")
    ap.add_argument("--out", default="bird_out")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()
    global TIMEOUT_S
    TIMEOUT_S = args.timeout

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.bird_dir, "dev.json"), encoding="utf-8") as f:
        items = json.load(f)

    stats = {"ours": Counter(), "baseline": Counter()}
    reasons = {"ours": Counter(), "baseline": Counter()}
    by_feature = {"ours": Counter(), "baseline": Counter()}
    feature_total = Counter()
    sensitive = Counter()
    conns: dict[str, tuple] = {}
    log = open(os.path.join(args.out, "bird_tenant.jsonl"), "w", encoding="utf-8")
    for i, item in enumerate(items):
        db = item["db_id"]
        if db not in conns:
            src = os.path.join(args.bird_dir, "dev_databases", db, f"{db}.sqlite")
            dst = os.path.join(args.out, f"{db}_tenants.sqlite")
            build_tenant_db(src, dst)
            full = connect(dst, None)
            conns[db] = (full, schema_of(full), {t: connect(dst, t) for t in TENANTS})
        full, schema, iso = conns[db]
        gold = item["SQL"]
        feats = features(gold)
        for t in TENANTS:
            feature_total.update(feats)
            rec = {"question_id": item.get("question_id", i), "db_id": db, "tenant": t, "gold": gold,
                   "features": feats}
            # Sensitivity: does the unscoped gold query on the full database differ from the
            # oracle? If it does not, this case cannot reveal a missing tenant filter.
            expected = run(iso[t], gold)
            unscoped = run(full, gold)
            rec["unscoped_differs"] = (
                None if isinstance(expected, str) or isinstance(unscoped, str) else unscoped != expected
            )
            sensitive[str(rec["unscoped_differs"])] += 1
            for name, fn in (("ours", verify_and_inject_ast), ("baseline", baseline_scope)):
                v = fn(gold, t, schema=schema)
                if not v.valid:
                    outcome = "rejected"
                    reasons[name][v.error.split(":")[0]] += 1
                else:
                    scoped = run(full, v.sql)
                    if isinstance(scoped, str) or isinstance(expected, str):
                        outcome = "error"
                        v.error = (  # which side failed and why ("interrupted" = time limit)
                            f"scoped: {scoped if isinstance(scoped, str) else 'ok'}; "
                            f"oracle: {expected if isinstance(expected, str) else 'ok'}"
                        )
                    else:
                        outcome = "isolated" if scoped == expected else "leaked"
                stats[name][outcome] += 1
                if outcome != "isolated":
                    by_feature[name].update(feats)
                prefix = "" if name == "ours" else "baseline_"
                rec |= {f"{prefix}outcome": outcome, f"{prefix}verified": v.sql, f"{prefix}error": v.error}
            log.write(json.dumps(rec) + "\n")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(items)} ours {dict(stats['ours'])} baseline {dict(stats['baseline'])}", flush=True)
    log.close()

    n = sum(stats["ours"].values())
    print(f"\n{len(items)} gold queries x {len(TENANTS)} tenants = {n} cases")
    print(f"unscoped gold query differs from the oracle (True = the case can reveal a missing filter): {dict(sensitive)}")
    for name in ("ours", "baseline"):
        print(f"[{name}]")
        for k in ("isolated", "leaked", "rejected", "error"):
            print(f"  {k:<9} {stats[name][k]:>5}  ({100 * stats[name][k] / n:.1f}%)")
        print("  rejection reasons:")
        for r, c in reasons[name].most_common():
            print(f"    {c:>5}  {r}")
    print("cases not isolated, by query feature (a case can have several):")
    for f, total in feature_total.most_common():
        print(f"  {f:<17} {total:>5} cases   ours {by_feature['ours'][f]:>4}   baseline {by_feature['baseline'][f]:>4}")


if __name__ == "__main__":
    main()
