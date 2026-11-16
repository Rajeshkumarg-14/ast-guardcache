"""
Compute every number used in the paper from files under results/.

    uv run python -X utf8 scripts/paper_numbers.py > results/paper_numbers.txt

Writes results/paper_numbers.json (read by docs/build/gen_nums.py). Inputs:
the end-to-end logs (<model>_run<k>.jsonl + .meta.json), the wide-schema logs
(wide_<model>.jsonl), rewriter_comparison.json, retriever_ablation.json,
cache_sweep.json, bird_tenant.jsonl, replay_pg.txt, pytest_report.txt,
pytest_postgres.txt and workload_validation.txt.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MODES = ["zero_shot", "hybrid_rag", "rag_normalize", "rag_ast", "ast_guardcache"]
MODELS = {"qwen2.5-coder-7b": "Qwen2.5-Coder 7B", "qwen3.5-4b": "Qwen3.5 4B", "qwen2.5-coder-1.5b": "Qwen2.5-Coder 1.5B"}
CATEGORIES = ["base", "repeat", "paraphrase", "near_miss", "repeated_const", "adversarial"]


def rows_of(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def mmm(xs: list[float]) -> dict:
    return {"mean": float(np.mean(xs)), "min": float(min(xs)), "max": float(max(xs)), "n": len(xs)}


def pct(rs: list[dict], key: str) -> float:
    return 100 * float(np.mean([bool(r[key]) for r in rs])) if rs else float("nan")


def model_stats(runs: list[list[dict]]) -> dict:
    out = {}
    for m in MODES:
        per_run = [[r for r in run if r["mode"] == m] for run in runs]
        pooled = [r for rs in per_run for r in rs]
        if not pooled:
            continue
        lat = [r["latency_ms"] for r in pooled]
        miss = [r for r in pooled if not r["cache_hit"]]
        hits = [r["latency_ms"] for r in pooled if r["cache_hit"]]
        out[m] = {
            "acc": mmm([pct(rs, "accurate") for rs in per_run]),
            "leak": mmm([pct(rs, "leaked") for rs in per_run]),
            "hit": mmm([pct(rs, "cache_hit") for rs in per_run]),
            "median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
            "tables": float(np.mean([r["pruned_tables"] for r in miss])) if miss else None,
            "prompt_tokens": float(np.mean([r.get("prompt_tokens", 0) for r in miss])) if miss else None,
            "retries": float(np.mean([r["retries"] for r in pooled])),
            "retried_requests": sum(r["retries"] > 0 for r in pooled),
            "errors": sum(r["error"] is not None for r in pooled),
            "empty_length": sum((not r["generated_sql"]) and r.get("finish_reason") == "length" for r in pooled),
            "hit_ms_mean": float(np.mean(hits)) if hits else None,
            "hit_ms_median": float(np.median(hits)) if hits else None,
            "hit_ms_range": [float(min(hits)), float(max(hits))] if hits else None,
            "model_call_median_ms": float(np.median([r["latency_ms"] for r in miss])) if miss else None,
            "leaks": dict(Counter(r["query_idx"] for r in pooled if r["leaked"])),
            "leak_qids": sorted({r["query_idx"] for r in pooled if r["leaked"]}),
            "leak_qcats": dict(Counter(c for _, c in {(r["query_idx"], r["category"]) for r in pooled if r["leaked"]})),
            "wrong_qids": sorted({r["query_idx"] for r in pooled if not r["accurate"]}),
            "wrong_any_run": len({r["query_idx"] for r in pooled if not r["accurate"]}),
            "error_qids": sorted({r["query_idx"] for r in pooled if r["error"]}),
            "wrong": dict(Counter(r["query_idx"] for r in pooled if not r["accurate"])),
            "by_category": {
                c: {"acc": float(np.mean([pct([r for r in rs if r["category"] == c], "accurate") for rs in per_run])),
                    "leak": float(np.mean([pct([r for r in rs if r["category"] == c], "leaked") for rs in per_run])),
                    "n": sum(r["category"] == c for r in per_run[0])}
                for c in CATEGORIES
            },
            "verify_ms": [r["verify_ms"] for r in pooled if r.get("verify_ms")],
        }
    return out


def bird() -> dict:
    path = os.path.join(RES, "bird_tenant.jsonl")
    if not os.path.exists(path):
        return {}
    rs = rows_of(path)
    out = {"cases": len(rs), "queries": len({(r["db_id"], r["question_id"]) for r in rs}),
           "dbs": len({r["db_id"] for r in rs})}
    for name, key in (("ours", "outcome"), ("baseline", "baseline_outcome")):
        out[name] = dict(Counter(r[key] for r in rs))
        errs = [r[key.replace("outcome", "error")] for r in rs if r[key] == "error"]
        out[name + "_error_kinds"] = dict(Counter("time limit" if "interrupted" in e else e[:80] for e in errs))
        out[name + "_error_queries"] = len({(r["db_id"], r["question_id"]) for r in rs if r[key] == "error"})
        bad = [r for r in rs if r[key] not in ("isolated",)]
        out[name + "_not_isolated_by_feature"] = dict(Counter(f for r in bad for f in r["features"]))
        out[name + "_by_outcome_feature"] = {
            o: dict(Counter(f for r in rs if r[key] == o for f in r["features"])) for o in ("leaked", "error", "rejected")}
    # cases in which the unscoped gold query differs from the oracle, i.e. that can reveal a missing filter
    out["sensitive_cases"] = sum(r.get("unscoped_differs") is True for r in rs)
    out["sensitive_by_tenant"] = dict(Counter(r["tenant"] for r in rs if r.get("unscoped_differs") is True))
    feats = Counter(f for r in rs if r["tenant"] == 1 for f in r["features"])
    out["query_features"] = dict(feats)
    return out


def similarities() -> dict:
    from app.cache import strip_entities
    from app.retriever import _get_model

    def cos(a, b):
        e = _get_model().encode([a, b], normalize_embeddings=True)
        return float(e[0] @ e[1])

    raw = cos("List CSE students with CGPA > 8.5", "List ECE students with CGPA > 9.0")
    m1 = cos(strip_entities("Show assessments with score > 80")[0], strip_entities("Show assessments with score < 80")[0])
    m2 = cos(strip_entities("Show the top 5 students by cgpa")[0], strip_entities("Show the bottom 5 students by cgpa")[0])
    return {"raw_constants": raw, "masked_gt_lt": m1, "masked_top_bottom": m2}


def workload() -> dict:
    from app.database import execute_sql, reset_connection
    from scripts.workload import BENCHMARK_QUERIES, seed_sql

    reset_connection(seed_sql())
    counts = {t: {r["tenant_id"]: r["n"] for r in execute_sql(f"SELECT tenant_id, COUNT(*) AS n FROM {t} GROUP BY tenant_id")}
              for t in ("students", "assessments", "hackathon_submissions", "billing_invoices", "webhook_events",
                        "notification_queue", "audit_logs")}
    val = open(os.path.join(RES, "workload_validation.txt"), encoding="utf-8").read()
    empty = re.search(r"empty or all-zero/NULL result: (\d+) \[(.*?)\]", val)
    return {"n": len(BENCHMARK_QUERIES), "by_category": dict(Counter(q["category"] for q in BENCHMARK_QUERIES)),
            "by_tenant": dict(Counter(q["tenant_id"] for q in BENCHMARK_QUERIES)), "rows": counts,
            "empty_gold": int(empty.group(1)), "empty_gold_ids": empty.group(2),
            "problems": int(re.search(r"problems: (\d+)", val).group(1))}


def parse_table(path: str) -> dict:
    """replay_pg.txt -> {config: {column: count}}"""
    if not os.path.exists(path):
        return {}
    lines = open(path, encoding="utf-8").read().splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("config"))
    cols = lines[head].split()[1:]
    out = {"cases": int(lines[0].split()[0])}
    for line in lines[head + 1:]:
        parts = line.split()
        if len(parts) == len(cols) + 1:
            out[parts[0]] = dict(zip(cols, map(int, parts[1:])))
    return out


def pytest_summary(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    text = open(path, encoding="utf-8").read()
    line = [x for x in text.splitlines() if re.search(r"\d+ passed", x)][-1]
    nums = {k: int(v) for v, k in re.findall(r"(\d+) (passed|skipped|failed|error)", line)}
    skips = {m.group(2): int(m.group(1)) for m in re.finditer(r"SKIPPED \[(\d+)\].*?\((\w+)\)", text)}
    return {"line": line.strip("= ").strip(), **nums, "skips": skips}


def meta() -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, "*.meta.json"))):
        out[os.path.basename(p).removesuffix(".meta.json")] = json.load(open(p, encoding="utf-8"))
    return out


def main() -> None:
    N: dict = {"models": {}, "wide": {}}
    for key in MODELS:
        paths = sorted(glob.glob(os.path.join(RES, f"{key}_run*.jsonl")))
        if paths:
            N["models"][key] = {"runs": len(paths), "stats": model_stats([rows_of(p) for p in paths])}
    for p in sorted(glob.glob(os.path.join(RES, "wide_*.jsonl"))):
        N["wide"][os.path.basename(p).removesuffix(".jsonl")] = model_stats([rows_of(p)])
    verify = [v for m in N["models"].values() for s in m["stats"].values() for v in s.pop("verify_ms")]
    for w in N["wide"].values():
        for s in w.values():
            s.pop("verify_ms")
    N["verify_ms_logs"] = {"median": float(np.median(verify)), "p95": float(np.percentile(verify, 95)),
                           "max": float(max(verify)), "n": len(verify)} if verify else {}
    for name in ("rewriter_comparison", "retriever_ablation", "cache_sweep"):
        p = os.path.join(RES, name + ".json")
        N[name] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    N["bird"] = bird()
    N["replay_pg"] = parse_table(os.path.join(RES, "replay_pg.txt"))
    N["pytest"] = pytest_summary(os.path.join(RES, "pytest_report.txt"))
    N["pytest_postgres"] = pytest_summary(os.path.join(RES, "pytest_postgres.txt"))
    N["similarity"] = similarities()
    N["workload"] = workload()
    N["meta"] = meta()
    json.dump(N, open(os.path.join(RES, "paper_numbers.json"), "w", encoding="utf-8"), indent=1)

    # Human-readable digest
    for key, m in N["models"].items():
        print(f"{MODELS[key]}: {m['runs']} run(s)")
        for mode, s in m["stats"].items():
            print(f"  {mode:<15} acc {s['acc']['mean']:5.1f} ({s['acc']['min']:.1f}-{s['acc']['max']:.1f})  "
                  f"leak {s['leak']['mean']:4.1f} ({s['leak']['min']:.1f}-{s['leak']['max']:.1f})  "
                  f"hit {s['hit']['mean']:4.1f}  med {s['median_ms']:7.0f}  p95 {s['p95_ms']:7.0f}  "
                  f"tables {s['tables']}  tok {s['prompt_tokens']}  retries {s['retries']:.2f}  "
                  f"errors {s['errors']}  leaks {s['leaks']}")
        for mode in ("zero_shot", "ast_guardcache"):
            s = m["stats"].get(mode)
            if s:
                print(f"  {mode} by category: " + ", ".join(
                    f"{c} {v['acc']:.1f}/{v['leak']:.1f}" for c, v in s["by_category"].items()))
    for key, w in N["wide"].items():
        print(f"{key}:")
        for mode, s in w.items():
            print(f"  {mode:<15} acc {s['acc']['mean']:5.1f} leak {s['leak']['mean']:4.1f} med {s['median_ms']:7.0f} "
                  f"tables {s['tables']} tok {s['prompt_tokens']} hit {s['hit']['mean']:.1f}")
    print("verifier time in logs:", N["verify_ms_logs"])
    print("bird:", json.dumps(N["bird"], indent=None)[:1500])
    print("replay_pg:", N["replay_pg"])
    print("pytest:", N["pytest"])
    print("pytest_postgres:", N["pytest_postgres"])
    print("similarity:", N["similarity"])
    print("workload:", N["workload"])


if __name__ == "__main__":
    main()
