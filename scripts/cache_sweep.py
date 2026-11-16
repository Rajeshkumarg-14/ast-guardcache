"""
Cache evaluation from the benchmark logs.

    uv run python -X utf8 scripts/cache_sweep.py results/<model>_run*.jsonl --out results/cache_sweep

1. End to end: per category, hit rate, hits that are correct, and false hits
   (a hit whose result is wrong while the same model, in the same run,
   answered the question correctly without the cache, i.e. in RAG + AST).
2. Offline replay, no LLM: the workload is replayed in order with the SQL each
   model wrote in RAG + AST held fixed. On a miss, that SQL is stored if it
   verified and ran (as the pipeline does); on a hit, the rebound SQL is
   verified, executed and scored against the gold query. Done for
   tau in {0.80, 0.84, 0.88, 0.92, 0.96} and for an exact-match cache that hits
   only on identical normalized question text (same rebinding and checks).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import ParameterizedSemanticCache, _bind_template, strip_entities
from app.database import execute_sql, reset_connection, tenant_only_connection
from app.verifier import verify_and_inject_ast
from scripts.run_benchmark import check_execution_accuracy, check_tenant_leak
from scripts.workload import BENCHMARK_QUERIES, CATEGORIES, seed_sql

TAUS = (0.80, 0.84, 0.88, 0.92, 0.96)


class ExactMatchCache(ParameterizedSemanticCache):
    """Hits only when the normalized question text is identical."""

    def lookup(self, query: str, tenant_id: int):
        normalised, params = strip_entities(query)
        for e in self._entries:
            if e.tenant_id == tenant_id and e.normalised_query == normalised and len(e.params) == len(params):
                rebound = _bind_template(e.sql_template, params, query)
                if rebound is not None:
                    return rebound, True
        return None, False


def load(path: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["mode"]][r["query_idx"]] = r
    return out


def end_to_end(logs: dict[str, dict]) -> dict:
    per = {c: Counter() for c in CATEGORIES}
    for runs in logs.values():
        g, ref = runs["ast_guardcache"], runs["rag_ast"]
        for i, r in g.items():
            c = per[r["category"]]
            c["requests"] += 1
            if r["cache_hit"]:
                c["hits"] += 1
                c["hits_correct"] += r["accurate"]
                c["false_hits"] += (not r["accurate"]) and ref[i]["accurate"]
                c["hits_leaked"] += r["leaked"]
    return {c: dict(v) for c, v in per.items()}


def replay(runs: dict[str, dict], cache: ParameterizedSemanticCache, iso) -> Counter:
    ref = runs["rag_ast"]
    out = Counter()
    for q in BENCHMARK_QUERIES:
        i, t = q["id"], q["tenant_id"]
        sql, hit = cache.lookup(q["question"], t)
        if hit:
            v = verify_and_inject_ast(sql, t)
            if v.valid:
                out["hits"] += 1
                out[f"hits:{q['category']}"] += 1
                try:
                    rows = execute_sql(v.sql)
                    ok = not check_tenant_leak(v.sql, iso[t]) and check_execution_accuracy(rows, q["gold_sql"])
                except Exception:  # noqa: BLE001
                    ok = False
                out["hits_correct"] += ok
                if not ok:
                    out["wrong_hits"] += 1
                    out[f"wrong:{q['category']}"] += 1
                    out["false_hits"] += ref[i]["accurate"]
                continue
        r = ref[i]
        if r["error"] is None and r["generated_sql"]:
            cache.store(q["question"], r["generated_sql"], t)
    out["entries"] = cache.size
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="results/cache_sweep")
    args = ap.parse_args()

    reset_connection(seed_sql())
    iso = {t: tenant_only_connection(t) for t in (1, 2)}
    logs = {os.path.basename(p).removesuffix(".jsonl"): load(p) for p in args.logs}

    e2e = end_to_end(logs)
    sweep: dict[str, dict] = {}
    configs = [(f"tau={tau:.2f}", lambda tau=tau: ParameterizedSemanticCache(threshold=tau)) for tau in TAUS]
    configs.append(("exact match", ExactMatchCache))
    for name, make in configs:
        total = Counter()
        for runs in logs.values():
            total += replay(runs, make(), iso)
        sweep[name] = dict(total)

    lines = [f"{len(logs)} logs: {', '.join(logs)}", "", "end to end (AST-GuardCache configuration), summed over logs:",
             f"  {'category':<15}{'requests':>9}{'hits':>6}{'correct':>9}{'false hits':>12}{'leaked':>8}"]
    for c in CATEGORIES:
        v = e2e[c]
        lines.append(f"  {c:<15}{v.get('requests', 0):>9}{v.get('hits', 0):>6}{v.get('hits_correct', 0):>9}"
                     f"{v.get('false_hits', 0):>12}{v.get('hits_leaked', 0):>8}")
    lines += ["", "offline replay of the RAG + AST SQL, summed over logs:",
              f"  {'cache':<12}{'hits':>6}{'correct':>9}{'wrong':>7}{'false':>7}  hits by category (wrong)"]
    for name, v in sweep.items():
        cats = ", ".join(f"{c} {v.get('hits:' + c, 0)} ({v.get('wrong:' + c, 0)})" for c in CATEGORIES
                         if v.get("hits:" + c))
        lines.append(f"  {name:<12}{v.get('hits', 0):>6}{v.get('hits_correct', 0):>9}{v.get('wrong_hits', 0):>7}"
                     f"{v.get('false_hits', 0):>7}  {cats}")
    json.dump({"logs": list(logs), "end_to_end": e2e, "sweep": sweep}, open(args.out + ".json", "w", encoding="utf-8"),
              indent=1)
    open(args.out + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
