"""
Retriever ablation without an LLM.

    uv run python -X utf8 scripts/retriever_ablation.py --out results/retriever_ablation

For every workload question and each ranker (BM25 only, dense only, RRF, RRF
plus one-hop foreign-key closure), on the 10-table schema and on the 60-table
wide schema, for K in {2, 4, 6}: whether all tables of the gold query are
retrieved, and how many tables go into the prompt. RRF + FK at K = 4 on the
10-table schema is checked against app.retriever.HybridSchemaRetriever.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sqlglot
from rank_bm25 import BM25Okapi
from sqlglot import exp

from app.config import settings
from app.database import FK_GRAPH, TABLE_DDLS, WIDE_DDLS, WIDE_FK_GRAPH
from app.retriever import HybridSchemaRetriever, _get_model, _tokenize
from scripts.workload import BENCHMARK_QUERIES

KS = (2, 4, 6)
METHODS = ("bm25", "dense", "rrf", "bm25_fk", "dense_fk", "rrf_fk")


def rankings(ddls: dict[str, str], questions: list[str]) -> dict[str, list[list[str]]]:
    names = list(ddls)
    bm25 = BM25Okapi([_tokenize(d) for d in ddls.values()])
    model = _get_model()
    emb = model.encode(list(ddls.values()), normalize_embeddings=True)
    out = {"bm25": [], "dense": [], "rrf": []}
    for q in questions:
        # one question at a time, exactly as the application encodes it
        qe = model.encode([q], normalize_embeddings=True)
        b = np.argsort(-bm25.get_scores(_tokenize(q)))
        d = np.argsort(-(emb @ qe.T).squeeze())
        score: dict[int, float] = {}
        for order in (b, d):
            for rank, idx in enumerate(order, start=1):
                score[int(idx)] = score.get(int(idx), 0.0) + 1.0 / (settings.RRF_K + rank)
        r = sorted(score, key=lambda i: score[i], reverse=True)
        out["bm25"].append([names[i] for i in b])
        out["dense"].append([names[i] for i in d])
        out["rrf"].append([names[i] for i in r])
    return out


def retrieved(method: str, ranks: dict, qi: int, k: int, fk: dict[str, list[str]]) -> set[str]:
    if not method.endswith("_fk"):
        return set(ranks[method][qi][:k])
    top = ranks[method.removesuffix("_fk")][qi][:k]  # the ranker's top K plus one-hop FK neighbours
    return set(top) | {n for t in top for n in fk.get(t, [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/retriever_ablation")
    args = ap.parse_args()

    questions = [q["question"] for q in BENCHMARK_QUERIES]
    gold = [{t.name.lower() for t in sqlglot.parse_one(q["gold_sql"], read="sqlite").find_all(exp.Table)}
            for q in BENCHMARK_QUERIES]
    schemas = {"10 tables": (dict(TABLE_DDLS), dict(FK_GRAPH)),
               "60 tables": (TABLE_DDLS | WIDE_DDLS, FK_GRAPH | WIDE_FK_GRAPH)}

    # The default setting must reproduce the application's retriever exactly
    app_retriever = HybridSchemaRetriever()
    ranks10 = rankings(schemas["10 tables"][0], questions)
    for qi, q in enumerate(questions):
        assert set(app_retriever.retrieve(q)[0]) == retrieved("rrf_fk", ranks10, qi, settings.RETRIEVER_TOP_K, FK_GRAPH)

    result = {}
    lines = [f"{len(questions)} questions; recall = share whose gold tables are all retrieved; "
             "tables = mean number of tables in the prompt"]
    for sname, (ddls, fk) in schemas.items():
        ranks = ranks10 if sname == "10 tables" else rankings(ddls, questions)
        result[sname] = {}
        lines.append(f"\n{sname}")
        lines.append(f"  {'method':<9}" + "".join(f"   K={k}: recall  tables" for k in KS))
        for m in METHODS:
            row = {}
            cells = []
            for k in KS:
                sets = [retrieved(m, ranks, qi, k, fk) for qi in range(len(questions))]
                hit = [g <= s for g, s in zip(gold, sets)]
                row[k] = {"recall": 100 * float(np.mean(hit)), "tables": float(np.mean([len(s) for s in sets])),
                          "missed": [i + 1 for i, h in enumerate(hit) if not h]}
                cells.append(f"   {row[k]['recall']:>11.1f}  {row[k]['tables']:>6.1f}")
            result[sname][m] = row
            lines.append(f"  {m:<9}" + "".join(cells))
    lines.append("\nquestions missed by rrf_fk at K=4:")
    for sname in schemas:
        lines.append(f"  {sname}: {result[sname]['rrf_fk'][4]['missed']}")
    json.dump(result, open(args.out + ".json", "w", encoding="utf-8"), indent=1)
    open(args.out + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
