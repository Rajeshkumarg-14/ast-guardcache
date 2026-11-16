"""
Concurrency / load tests (mock LLM, offline).

1. 2,000 pipeline runs from 16 threads: no errors, no cross-tenant leaks
   (differential oracle), cache stays tenant-partitioned.
2. 20,000 verifier calls from 16 threads: output is deterministic per input.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor

from app.database import execute_sql, tenant_only_connection
from app.pipeline import Pipeline
from app.verifier import verify_and_inject_ast

QUESTIONS = [
    "Show students in cse", "students with cgpa > 8", "total students", "average cgpa",
    "hackathon results", "assessment report", "billing summary", "overdue payments",
    "notification backlog", "webhook failures", "audit trail", "top scorer",
]


from app.database import canon_rows as _canon  # noqa: E402


def test_concurrent_pipeline_no_errors_no_leaks():
    pipeline = Pipeline(use_mock_llm=True)
    jobs = list(itertools.islice(
        itertools.cycle(itertools.product(QUESTIONS, (1, 2), ("ast_guardcache", "rag_ast"))),
        2000,
    ))

    def run(job):
        q, t, mode = job
        return t, pipeline.run(question=q, tenant_id=t, mode=mode)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(run, jobs))

    assert all(r.error is None for _, r in results), [r.error for _, r in results if r.error][:3]
    assert sum(r.cache_hit for _, r in results) > 0

    iso = {t: tenant_only_connection(t) for t in (1, 2)}
    for t, sql in {(t, r.verified_sql) for t, r in results}:
        assert _canon(execute_sql(sql)) == _canon(execute_sql(sql, conn=iso[t])), sql

    assert all(e.tenant_id in (1, 2) for e in pipeline.cache._entries)


def test_concurrent_verifier_is_deterministic():
    queries = [
        "SELECT * FROM students",
        "SELECT s.name, a.score FROM students s LEFT JOIN assessments a ON s.id = a.student_id",
        "SELECT name, (SELECT AVG(cgpa) FROM students) AS g FROM students",
        "WITH c AS (SELECT * FROM billing_invoices) SELECT COUNT(*) FROM c",
        "SELECT id FROM tenants UNION SELECT tenant_id FROM students",
    ]
    jobs = list(itertools.islice(itertools.cycle(itertools.product(queries, (1, 2))), 20000))
    expected = {job: verify_and_inject_ast(*job).sql for job in set(jobs)}

    with ThreadPoolExecutor(max_workers=16) as pool:
        outputs = list(pool.map(lambda job: (job, verify_and_inject_ast(*job)), jobs))

    for job, result in outputs:
        assert result.valid and result.sql == expected[job]
