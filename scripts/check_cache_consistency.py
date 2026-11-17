"""
Check that the cache decisions recorded in the end-to-end logs are the ones the
current cache code makes, without calling a model.

    uv run python -X utf8 scripts/check_cache_consistency.py results/<model>_run*.jsonl

For each log, the AST-GuardCache requests are replayed in order: lookup with the
current code; on a miss, the SQL the model wrote in that request is stored if
the request executed (as the pipeline does). The script reports every request
whose hit/miss decision or rebuilt SQL differs from the log. Zero differences
means a change to the cache rules did not invalidate the recorded runs.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import ParameterizedSemanticCache
from app.verifier import verify_and_inject_ast


def main(paths: list[str]) -> int:
    total = 0
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        rows = sorted((r for r in rows if r["mode"] == "ast_guardcache"), key=lambda r: r["query_idx"])
        cache = ParameterizedSemanticCache()
        diffs = []
        for r in rows:
            sql, hit = cache.lookup(r["question"], r["tenant_id"])
            if hit:
                check = verify_and_inject_ast(sql, r["tenant_id"])
                hit = check.valid
            if hit != r["cache_hit"]:
                diffs.append(f"Q{r['query_idx']}: logged hit={r['cache_hit']}, now hit={hit}")
            elif hit and check.sql != r["executed_sql"]:
                diffs.append(f"Q{r['query_idx']}: hit in both, rebuilt SQL differs")
            if not r["cache_hit"] and r["error"] is None and r["generated_sql"]:
                cache.store(r["question"], r["generated_sql"], r["tenant_id"])
        print(f"{os.path.basename(path)}: {len(rows)} requests, {sum(r['cache_hit'] for r in rows)} logged hits, "
              f"{len(diffs)} differences")
        for d in diffs:
            print("  " + d)
        total += len(diffs)
    print(f"total differences: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
