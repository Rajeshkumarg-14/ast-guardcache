"""
Re-score benchmark logs from the SQL they recorded, without calling a model.

    uv run python -X utf8 scripts/rescore_logs.py results/<model>_run*.jsonl
    WIDE_SCHEMA=1 uv run python -X utf8 scripts/rescore_logs.py results/wide_*.jsonl

Scoring is a deterministic function of the executed SQL and the benchmark
database, so it can be repeated after a change to the scoring code. This was
needed for three scoring fixes. Result sets used to be compared with exact float equality, and an
AVG over the same rows can differ in its last digits when the engine visits
the rows in another order. Results are now compared to 12 significant digits
(app.database.canon_rows). An empty model output used to score as correct when
the gold result was empty; it now counts as not executed. Two result columns
with the same name used to collapse into one; they are now kept apart.

For every logged request that executed, the SQL is run again and `leaked` and
`accurate` are recomputed. The script prints every flag that changes, keeps
the original file under results/superseded/pre_float_fix/ and rewrites the log
only if something changed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import execute_sql, reset_connection, tenant_only_connection
from scripts.run_benchmark import check_execution_accuracy, check_tenant_leak, executed
from scripts.workload import BENCHMARK_QUERIES, seed_sql


def main(paths: list[str]) -> int:
    reset_connection(seed_sql())
    iso = {t: tenant_only_connection(t) for t in (1, 2)}
    total = 0
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        changed = []
        for r in rows:
            q = BENCHMARK_QUERIES[r["query_idx"] - 1]
            assert q["question"] == r["question"], "log does not match the current workload"
            if not executed(r["executed_sql"], r["error"]):
                # no SQL or an error: inaccurate and not a leak, whatever the gold result is
                if r["leaked"] or r["accurate"]:
                    changed.append((r["query_idx"], r["mode"], r["category"], r["leaked"], False, r["accurate"], False))
                    r["leaked"] = r["accurate"] = False
                continue
            try:
                result = execute_sql(r["executed_sql"])
            except Exception as exc:  # executed during the run, so this would be a real problem
                print(f"  {os.path.basename(path)} Q{r['query_idx']} {r['mode']}: no longer executes: {exc}")
                return 1
            leaked = check_tenant_leak(r["executed_sql"], iso[r["tenant_id"]])
            accurate = (not leaked) and check_execution_accuracy(result, q["gold_sql"])
            if (leaked, accurate) != (r["leaked"], r["accurate"]):
                changed.append((r["query_idx"], r["mode"], r["category"], r["leaked"], leaked, r["accurate"], accurate))
                r["leaked"], r["accurate"] = leaked, accurate
        print(f"{os.path.basename(path)}: {len(rows)} rows, {len(changed)} flags changed")
        for idx, mode, cat, l0, l1, a0, a1 in changed:
            print(f"  Q{idx} {mode} ({cat}): leaked {l0} -> {l1}, accurate {a0} -> {a1}")
        if changed:
            keep = os.path.join(os.path.dirname(path), "superseded", "pre_float_fix")
            os.makedirs(keep, exist_ok=True)
            shutil.copy2(path, keep)
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        total += len(changed)
    print(f"total flags changed: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
