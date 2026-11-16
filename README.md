# AST-GuardCache

> **Schema-Aware Hybrid Retrieval and Parameterized Template Caching for Tenant-Isolated Text-to-SQL**

Code, tests, benchmark harness and result logs for the paper, plus an interactive web prototype.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Natural Language Query                       │
│                    + tenant_id + mode flag                       │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐     ┌──────────────────────────┐
│  Parameterized Semantic     │────▶│  Cache HIT? Rebind       │
│  Cache (Cosine ≥ 0.88)     │     │  params → Skip LLM       │
└─────────────┬───────────────┘     └──────────────────────────┘
              │ MISS
              ▼
┌─────────────────────────────┐
│  Hybrid Schema Retriever    │
│  BM25 + Dense (RRF)        │
│  + 1-hop FK Closure         │
└─────────────┬───────────────┘
              │ Pruned DDLs
              ▼
┌─────────────────────────────┐
│  LLM (OpenRouter)           │
│  Text-to-SQL Generation     │
└─────────────┬───────────────┘
              │ Candidate SQL
              ▼
┌─────────────────────────────┐     ┌──────────────────────────┐
│  Symbolic AST Verifier      │────▶│  Self-correction retry   │
│  (sqlglot)                  │     │  loop (max 2 retries)    │
│  • Read-only guard          │     └──────────────────────────┘
│  • Schema existence check   │
│  • Per-table tenant scoping │
└─────────────┬───────────────┘
              │ Verified SQL
              ▼
┌─────────────────────────────┐
│  SQLite Execution           │
│  + Telemetry Response       │
└─────────────────────────────┘
```

## Quick Start

### 1. Clone & Install

```bash
cd ast-guardcache
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 2. Run Tests (offline once the embedding model is in the local Hugging Face cache)

```bash
uv run pytest -v
```

### 3. Run Benchmark (Mock LLM)

```bash
uv run python scripts/run_benchmark.py --mock-llm
```

### 4. Run Benchmark (real model)

Any OpenAI-compatible endpoint works. The paper's results use local models served by
[Ollama](https://ollama.com), which costs nothing:

```bash
ollama pull qwen2.5-coder:7b
export OPENROUTER_BASE_URL=http://127.0.0.1:11434/v1 OPENROUTER_API_KEY=ollama
OPENROUTER_MODEL=qwen2.5-coder:7b uv run python -X utf8 scripts/run_benchmark.py --log results/qwen2.5-coder-7b_run1.jsonl
```

Leave `OPENROUTER_BASE_URL` unset to use OpenRouter with the key in `.env`. The benchmark
aborts on provider errors (no credits, rate limits, a dropped connection) instead of scoring
them as model failures. It overwrites `benchmark_results.csv` and `ieee_benchmark_chart.pdf`.

### Workload

`scripts/workload.py` holds the benchmark workload: 102 questions with hand-written gold SQL
and a category (`base`, `repeat`, `paraphrase`, `near_miss`, `repeated_const`, `adversarial`),
and a deterministic generator of extra seed rows (about 40 students per tenant plus matching
rows in the other tables) that is loaded only by the benchmark
(`reset_connection(extra_sql=seed_sql())`). The default database and the tests use the small
base seed. The gold queries are checked before any model run:

```bash
uv run python -X utf8 scripts/validate_workload.py > results/workload_validation.txt
```

### Experiments behind the paper (all outputs in `results/`)

```bash
# End-to-end: 3 runs each for qwen2.5-coder 7B and 1.5B, 1 run for qwen3.5 4B
OPENROUTER_MODEL=qwen2.5-coder:7b uv run python -X utf8 scripts/run_benchmark.py --log results/qwen2.5-coder-7b_run1.jsonl
# (repeat for run2/run3 and the other models)

# Wide schema (50 empty distractor tables, opt-in) with a 16k-context copy of the model
ollama create qwen2.5-coder-7b-16k -f scripts/ollama/qwen2.5-coder-7b-16k.Modelfile
WIDE_SCHEMA=1 OPENROUTER_MODEL=qwen2.5-coder-7b-16k:latest uv run python -X utf8 scripts/run_benchmark.py \
    --modes zero_shot,hybrid_rag,ast_guardcache --log results/wide_qwen2.5-coder-7b.jsonl

# No LLM needed:
uv run python -X utf8 scripts/compare_rewriters.py       # ours vs outer-WHERE baseline, generated queries
uv run python -X utf8 scripts/retriever_ablation.py      # BM25 / dense / RRF / RRF+FK, 10 and 60 tables
uv run python -X utf8 scripts/cache_sweep.py results/qwen2.5-coder-*_run*.jsonl results/qwen3.5-4b_run*.jsonl
uv run python -X utf8 -u scripts/bird_tenant_eval.py <bird>/dev_20240627 --out <bird_out>   # ours and baseline

# PostgreSQL: verifier vs Row-Level Security vs both (local container, host-only port)
docker compose -f docker-compose.research.yml up -d
uv run pytest tests/test_postgres.py -q -rs
uv run python scripts/replay_pg.py results/*_run*.jsonl > results/replay_pg.txt
docker compose -f docker-compose.research.yml stop

# Re-score logs from their recorded SQL after a change to the scoring code (no model calls).
# Result sets are compared order-insensitively and floats to 12 significant digits
# (app.database.canon_rows); results/rescore_report.txt records the result (7 Qwen3.5 accuracy flags corrected for empty output).
uv run python -X utf8 scripts/rescore_logs.py results/qwen*_run*.jsonl
WIDE_SCHEMA=1 uv run python -X utf8 scripts/rescore_logs.py results/wide_*.jsonl

# Every number in the paper, from the files above
uv run python -X utf8 scripts/paper_numbers.py > results/paper_numbers.txt
uv run python scripts/make_figures.py results/qwen2.5-coder-7b_run*.jsonl --out docs/figures
```

Each benchmark log has a `.meta.json` (model digest, Ollama version, endpoint, library
versions, embedding device). `scripts/baseline_rewriter.py` is the scoping step of the first
version of our verifier (git commit `a2624d3`), kept only as a baseline.

**The API is a demo.** `/api/query` takes `tenant_id` from the request body and has no
authentication; a deployment must take the tenant from the authenticated session. With
`APP_ENV=production` only the verified modes (`rag_ast`, `ast_guardcache`) are accepted.

### 5. Launch Web Dashboard

```bash
# With uv
uv run uvicorn app.main:app --reload --port 8000

# With Docker
docker compose up --build
```

Then open **http://localhost:8000**

## Ablation Modes

| Mode | RAG | AST Verify | Cache | Description |
|------|-----|------------|-------|-------------|
| `zero_shot` | ✗ | ✗ | ✗ | Full schema → LLM → raw execute |
| `hybrid_rag` | ✓ | ✗ | ✗ | Pruned schema → LLM → raw execute |
| `rag_normalize` | ✓ | ✗ | ✗ | As `hybrid_rag`, SQL re-printed by sqlglot (separates syntax repair from verification) |
| `rag_ast` | ✓ | ✓ | ✗ | Pruned schema → LLM → AST verify + tenant scoping |
| `ast_guardcache` | ✓ | ✓ | ✓ | **Full proposed pipeline** |

## Schema (10 Tables)

Multi-tenant EdTech/FinTech schema modeled after the Stride production system:

- `tenants` — Multi-tenant registry
- `students` — Student records with department, CGPA
- `competency_tracks` — Learning tracks and categories
- `assessments` — Student assessment scores
- `hackathons` — Hackathon events
- `hackathon_submissions` — Student hackathon submissions
- `audit_logs` — System audit trail
- `billing_invoices` — Billing and invoicing
- `webhook_events` — Webhook delivery tracking
- `notification_queue` — Notification delivery queue

## What the verifier enforces

1. **Tenant scoping**: every reference to a table, in every scope (joins, subqueries, CTEs,
   set operations), is replaced by `(SELECT * FROM t WHERE tenant_id = <caller>) AS t`.
   This also runs on cache hits.
2. **Read-only**: only queries pass; DML/DDL anywhere in the tree, unknown functions,
   system catalogs and schema-qualified tables are rejected. The SQLite connection is also
   opened `query_only`.
3. **Catalog check**: tables and qualified columns that do not exist are rejected with a
   message the model can use to correct itself (max 2 retries). Columns of CTEs and derived
   tables are not resolved; such mistakes are left to the database.

The verifier enforces tenant scoping on the queries it accepts; it does not check that a query
is correct, and isolation is tested (see below), not proved: a valid, well-scoped query that
answers the wrong question still passes. The test suite checks isolation with a differential oracle
(scoped query on the full database must equal the original query on a copy that holds only
the caller's rows).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Healthcheck |
| `POST` | `/api/query` | Run text-to-SQL pipeline |
| `POST` | `/api/benchmark` | Trigger ablation benchmark |
| `GET` | `/` | Interactive dashboard |

## License

Research use only.
