# AST-GuardCache

> **Schema-Aware Hybrid RAG and Parameterized Semantic Caching Architecture for Low-Latency, Zero-Hallucination Text-to-SQL**

IEEE Research Paper Empirical Artifacts & Interactive Web Prototype.

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
│  • Tenant RLS injection     │
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

### 2. Run Tests (100% Offline)

```bash
uv run pytest -v
```

### 3. Run Benchmark (Mock LLM)

```bash
uv run python scripts/run_benchmark.py --mock-llm
```

### 4. Run Benchmark (Live LLM via OpenRouter)

```bash
uv run python scripts/run_benchmark.py
```

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
| `rag_ast` | ✓ | ✓ | ✗ | Pruned schema → LLM → AST verify + tenant inject |
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

## Key Security Guarantees

1. **Zero Data Leakage**: AST verifier injects `WHERE tenant_id = ?` into every tenant-scoped table
2. **No SQL Injection**: Only SELECT statements pass the read-only guard
3. **No Hallucinations**: Schema existence guard blocks queries referencing non-existent tables/columns
4. **Self-Correction**: LLM errors are fed back for automatic retry (max 2 attempts)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Healthcheck |
| `POST` | `/api/query` | Run text-to-SQL pipeline |
| `POST` | `/api/benchmark` | Trigger ablation benchmark |
| `GET` | `/` | Interactive dashboard |

## License

Research use only. © 2025
