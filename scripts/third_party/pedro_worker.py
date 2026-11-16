"""
Runs the published rewriter of Pedro et al. unchanged, under the sqlglot version
its artifact pins (20.10.0), in a separate interpreter:

    uv run --no-project --with sqlglot==20.10.0 python -I scripts/third_party/pedro_worker.py

Reads one JSON object per line on stdin: {"sql": ..., "clauses": [{"table_name", "where_clause"}]}.
Writes one per line on stdout: {"sql": rewritten} or {"error": message}, plus "ms".

p2sql_query_rewriter.py is RQ4/langshield-langchain/langshield/sql_query_rewriter.py from
https://github.com/rodrigo-pedro/P2SQL (MIT; artifact https://zenodo.org/records/14743615).
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlglot  # noqa: E402
from p2sql_query_rewriter import QueryRewritter  # noqa: E402

assert sqlglot.__version__.startswith("20.10"), sqlglot.__version__

for line in sys.stdin:
    req = json.loads(line)
    t0 = time.perf_counter()
    try:
        out = {"sql": QueryRewritter(req["clauses"])(req["sql"])}
    except Exception as exc:  # the rewriter's own failure is an outcome, not a crash
        out = {"error": f"{type(exc).__name__}: {exc}"}
    out["ms"] = (time.perf_counter() - t0) * 1000
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
