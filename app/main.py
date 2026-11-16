"""
AST-GuardCache — FastAPI Web Application & Interactive Dashboard.

Endpoints:
- GET  /health        → Healthcheck
- POST /api/query     → Run the text-to-SQL pipeline
- POST /api/benchmark → Trigger the ablation benchmark
- GET  /              → Serve the interactive dashboard
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.database import execute_sql
from app.pipeline import Pipeline, PipelineResult

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AST-GuardCache",
    description=(
        "Schema-aware hybrid retrieval and parameterized template caching "
        "for tenant-isolated text-to-SQL (research prototype)"
    ),
    version="0.1.0",
)

# Initialise the pipeline (with real LLM by default; tests can override)
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(use_mock_llm=False)
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """
    Demo request. ``tenant_id`` is taken from the body because this prototype
    has no authentication; a deployment must take it from the authenticated
    session instead, never from the client.
    """

    question: str
    tenant_id: int = 1
    mode: str = "ast_guardcache"


# Baseline modes execute unverified SQL; they exist for the ablation only
_VERIFIED_MODES = {"rag_ast", "ast_guardcache"}


class QueryResponse(BaseModel):
    mode: str
    question: str
    generated_sql: str
    verified_sql: str
    execution_rows: list[dict[str, Any]]
    cache_hit: bool
    ast_retries: int
    pruned_tables_count: int
    total_tables_count: int
    latency_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Healthcheck endpoint."""
    rows = execute_sql("SELECT COUNT(*) AS cnt FROM tenants")
    return {"status": "healthy", "tenants_count": rows[0]["cnt"] if rows else 0}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run the text-to-SQL pipeline (sync: FastAPI runs it in its threadpool)."""
    if settings.APP_ENV == "production" and req.mode not in _VERIFIED_MODES:
        raise HTTPException(status_code=403, detail="Unverified modes are disabled in production.")
    pipeline = get_pipeline()
    result: PipelineResult = pipeline.run(
        question=req.question,
        tenant_id=req.tenant_id,
        mode=req.mode,
    )
    return QueryResponse(
        mode=result.mode,
        question=result.question,
        generated_sql=result.generated_sql,
        verified_sql=result.verified_sql,
        execution_rows=result.execution_rows,
        cache_hit=result.cache_hit,
        ast_retries=result.ast_retries,
        pruned_tables_count=result.pruned_tables_count,
        total_tables_count=result.total_tables_count,
        latency_ms=round(result.latency_ms, 2),
        error=result.error,
    )


@app.post("/api/benchmark")
async def benchmark():
    """Trigger the ablation benchmark (mock LLM) and return summary."""
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/run_benchmark.py", "--mock-llm"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "status": "completed" if proc.returncode == 0 else "error",
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "stderr": proc.stderr[-1000:] if proc.stderr else "",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Dashboard (Single-page HTML)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AST-GuardCache — Interactive Text-to-SQL Dashboard</title>
    <meta name="description" content="Interactive prototype for the IEEE paper: AST-GuardCache — Schema-Aware Hybrid RAG and Parameterized Semantic Caching for Low-Latency, Zero-Hallucination Text-to-SQL." />
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'system-ui', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        surface: { 50: '#f8fafc', 100: '#0f172a', 200: '#1e293b', 300: '#334155' },
                        accent: { 400: '#818cf8', 500: '#6366f1', 600: '#4f46e5' },
                        emerald: { 400: '#34d399', 500: '#10b981' },
                        amber: { 400: '#fbbf24', 500: '#f59e0b' },
                        rose: { 400: '#fb7185', 500: '#f43f5e' },
                    }
                }
            }
        }
    </script>
    <style>
        body { font-family: 'Inter', system-ui, sans-serif; }
        .glass { background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(20px); border: 1px solid rgba(99, 102, 241, 0.15); }
        .glass-light { background: rgba(30, 41, 59, 0.35); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06); }
        .glow { box-shadow: 0 0 40px rgba(99, 102, 241, 0.15), 0 0 80px rgba(99, 102, 241, 0.05); }
        .glow-sm { box-shadow: 0 0 20px rgba(99, 102, 241, 0.12); }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse-glow { 0%, 100% { box-shadow: 0 0 8px rgba(99, 102, 241, 0.3); } 50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.6); } }
        .animate-fade-in-up { animation: fadeInUp 0.5s ease-out both; }
        .animate-pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
        .sql-box { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 1.6; }
        .badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 10px; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 3px; }
        .spinner { width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.2); border-top-color: #818cf8; border-radius: 50%; animation: spin 0.6s linear infinite; display: inline-block; }
        @keyframes spin { to { transform: rotate(360deg); } }
        table th { position: sticky; top: 0; z-index: 1; }
    </style>
</head>
<body class="bg-surface-100 text-slate-200 min-h-screen antialiased">
    <!-- Gradient orb decorations -->
    <div class="fixed inset-0 pointer-events-none overflow-hidden">
        <div class="absolute -top-40 -right-40 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl"></div>
        <div class="absolute -bottom-40 -left-40 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl"></div>
    </div>

    <div class="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Header -->
        <header class="text-center mb-10 animate-fade-in-up">
            <div class="inline-flex items-center gap-2 mb-3">
                <div class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></div>
                <span class="text-xs font-medium text-slate-400 tracking-widest uppercase">IEEE Research Prototype</span>
            </div>
            <h1 class="text-3xl sm:text-4xl lg:text-5xl font-extrabold bg-gradient-to-r from-accent-400 via-white to-emerald-400 bg-clip-text text-transparent leading-tight">
                AST-GuardCache
            </h1>
            <p class="mt-3 text-sm sm:text-base text-slate-400 max-w-2xl mx-auto leading-relaxed">
                Schema-Aware Hybrid RAG &amp; Parameterized Semantic Caching for
                <span class="text-white font-semibold">Tenant-Isolated</span> Text-to-SQL
            </p>
        </header>

        <!-- Query Panel -->
        <section class="glass rounded-2xl p-6 sm:p-8 glow mb-8 animate-fade-in-up" style="animation-delay: 0.1s;">
            <div class="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
                <!-- Question -->
                <div class="md:col-span-6">
                    <label for="query-input" class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Natural Language Query</label>
                    <input id="query-input" type="text" placeholder="e.g. Show all students in CSE with cgpa > 8.5"
                        class="w-full bg-surface-100/80 border border-slate-600/50 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-accent-500/50 focus:border-accent-500 transition" />
                </div>
                <!-- Tenant -->
                <div class="md:col-span-2">
                    <label for="tenant-select" class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Tenant</label>
                    <select id="tenant-select"
                        class="w-full bg-surface-100/80 border border-slate-600/50 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-accent-500/50 transition appearance-none cursor-pointer">
                        <option value="1">Tenant 1 — Stride</option>
                        <option value="2">Tenant 2 — Nova</option>
                    </select>
                </div>
                <!-- Mode -->
                <div class="md:col-span-2">
                    <label for="mode-select" class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Pipeline Mode</label>
                    <select id="mode-select"
                        class="w-full bg-surface-100/80 border border-slate-600/50 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-accent-500/50 transition appearance-none cursor-pointer">
                        <option value="ast_guardcache">AST-GuardCache (Full)</option>
                        <option value="rag_ast">RAG + AST</option>
                        <option value="hybrid_rag">Hybrid RAG Only</option>
                        <option value="zero_shot">Zero-Shot</option>
                    </select>
                </div>
                <!-- Submit -->
                <div class="md:col-span-2">
                    <button id="submit-btn" onclick="submitQuery()"
                        class="w-full bg-gradient-to-r from-accent-500 to-accent-600 hover:from-accent-400 hover:to-accent-500 text-white font-semibold rounded-xl px-4 py-3 text-sm transition-all duration-200 hover:shadow-lg hover:shadow-accent-500/25 active:scale-[0.98] flex items-center justify-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                        Execute
                    </button>
                </div>
            </div>
            <!-- Quick examples -->
            <div class="mt-4 flex flex-wrap gap-2">
                <span class="text-xs text-slate-500">Try:</span>
                <button onclick="setQuery('Show all students in CSE with cgpa > 8.5')" class="text-xs text-accent-400 hover:text-accent-300 bg-accent-500/10 px-3 py-1 rounded-full transition hover:bg-accent-500/20">Students in CSE</button>
                <button onclick="setQuery('List hackathon submissions with scores')" class="text-xs text-accent-400 hover:text-accent-300 bg-accent-500/10 px-3 py-1 rounded-full transition hover:bg-accent-500/20">Hackathon Scores</button>
                <button onclick="setQuery('Show pending billing invoices')" class="text-xs text-accent-400 hover:text-accent-300 bg-accent-500/10 px-3 py-1 rounded-full transition hover:bg-accent-500/20">Billing</button>
                <button onclick="setQuery('Show assessment scores with student names and tracks')" class="text-xs text-accent-400 hover:text-accent-300 bg-accent-500/10 px-3 py-1 rounded-full transition hover:bg-accent-500/20">Assessments</button>
                <button onclick="setQuery('Count total active students')" class="text-xs text-accent-400 hover:text-accent-300 bg-accent-500/10 px-3 py-1 rounded-full transition hover:bg-accent-500/20">Active Count</button>
            </div>
        </section>

        <!-- Results -->
        <div id="results-container" class="hidden space-y-6">
            <!-- Telemetry Cards -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 animate-fade-in-up" style="animation-delay: 0.15s;" id="telemetry-cards">
                <div class="glass-light rounded-xl p-4 text-center">
                    <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">Latency</p>
                    <p class="text-2xl font-bold text-white" id="stat-latency">—</p>
                    <p class="text-xs text-slate-500">ms</p>
                </div>
                <div class="glass-light rounded-xl p-4 text-center">
                    <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">Cache</p>
                    <p class="text-lg font-bold" id="stat-cache">—</p>
                </div>
                <div class="glass-light rounded-xl p-4 text-center">
                    <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">Schema Pruned</p>
                    <p class="text-2xl font-bold text-white"><span id="stat-pruned">—</span><span class="text-sm text-slate-400"> / <span id="stat-total">10</span></span></p>
                    <p class="text-xs text-slate-500">tables</p>
                </div>
                <div class="glass-light rounded-xl p-4 text-center">
                    <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">AST Retries</p>
                    <p class="text-2xl font-bold text-white" id="stat-retries">—</p>
                </div>
            </div>

            <!-- SQL Comparison -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in-up" style="animation-delay: 0.2s;">
                <div class="glass-light rounded-xl overflow-hidden">
                    <div class="px-5 py-3 border-b border-slate-700/50 flex items-center justify-between">
                        <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Raw LLM SQL</h3>
                        <span class="badge bg-amber-500/20 text-amber-400">Generated</span>
                    </div>
                    <pre class="p-5 sql-box text-amber-300/90 overflow-x-auto whitespace-pre-wrap" id="sql-raw">—</pre>
                </div>
                <div class="glass-light rounded-xl overflow-hidden glow-sm">
                    <div class="px-5 py-3 border-b border-slate-700/50 flex items-center justify-between">
                        <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">AST-Verified Tenant-Safe SQL</h3>
                        <span class="badge bg-emerald-500/20 text-emerald-400">Verified</span>
                    </div>
                    <pre class="p-5 sql-box text-emerald-300/90 overflow-x-auto whitespace-pre-wrap" id="sql-verified">—</pre>
                </div>
            </div>

            <!-- Error display -->
            <div id="error-box" class="hidden glass-light rounded-xl p-5 border-l-4 border-rose-500 animate-fade-in-up">
                <div class="flex items-start gap-3">
                    <svg class="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"/></svg>
                    <p class="text-sm text-rose-300" id="error-text"></p>
                </div>
            </div>

            <!-- Results Table -->
            <div class="glass-light rounded-xl overflow-hidden animate-fade-in-up" style="animation-delay: 0.25s;">
                <div class="px-5 py-3 border-b border-slate-700/50 flex items-center justify-between">
                    <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Query Results</h3>
                    <span class="text-xs text-slate-500" id="row-count"></span>
                </div>
                <div class="overflow-x-auto max-h-96">
                    <table class="w-full text-sm" id="results-table">
                        <thead id="results-thead" class="bg-surface-200/80 text-slate-400 text-xs uppercase tracking-wider"></thead>
                        <tbody id="results-tbody" class="divide-y divide-slate-700/30"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Loading overlay -->
        <div id="loading-overlay" class="hidden fixed inset-0 bg-surface-100/60 backdrop-blur-sm flex items-center justify-center z-50">
            <div class="glass rounded-2xl p-8 flex flex-col items-center gap-4">
                <div class="spinner" style="width:32px;height:32px;border-width:3px;"></div>
                <p class="text-sm text-slate-300">Running pipeline…</p>
            </div>
        </div>

        <!-- Footer -->
        <footer class="mt-16 text-center text-xs text-slate-600">
            <p>AST-GuardCache &middot; IEEE Research Prototype &middot; 2025</p>
        </footer>
    </div>

    <script>
        function setQuery(q) {
            document.getElementById('query-input').value = q;
        }

        async function submitQuery() {
            const question = document.getElementById('query-input').value.trim();
            if (!question) return;

            const tenantId = parseInt(document.getElementById('tenant-select').value);
            const mode = document.getElementById('mode-select').value;

            // Show loading
            document.getElementById('loading-overlay').classList.remove('hidden');
            document.getElementById('results-container').classList.add('hidden');
            document.getElementById('error-box').classList.add('hidden');

            try {
                const resp = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, tenant_id: tenantId, mode }),
                });
                const data = await resp.json();
                renderResults(data);
            } catch (err) {
                document.getElementById('error-box').classList.remove('hidden');
                document.getElementById('error-text').textContent = err.message;
                document.getElementById('results-container').classList.remove('hidden');
            } finally {
                document.getElementById('loading-overlay').classList.add('hidden');
            }
        }

        function renderResults(data) {
            document.getElementById('results-container').classList.remove('hidden');

            // Telemetry
            document.getElementById('stat-latency').textContent = data.latency_ms?.toFixed(1) ?? '—';
            document.getElementById('stat-pruned').textContent = data.pruned_tables_count ?? '—';
            document.getElementById('stat-total').textContent = data.total_tables_count ?? '10';
            document.getElementById('stat-retries').textContent = data.ast_retries ?? '0';

            const cacheEl = document.getElementById('stat-cache');
            if (data.cache_hit) {
                cacheEl.innerHTML = '<span class="badge bg-emerald-500/20 text-emerald-400 text-base">HIT</span>';
            } else {
                cacheEl.innerHTML = '<span class="badge bg-slate-600/40 text-slate-400 text-base">MISS</span>';
            }

            // SQL
            document.getElementById('sql-raw').textContent = data.generated_sql || '—';
            document.getElementById('sql-verified').textContent = data.verified_sql || '—';

            // Error
            if (data.error) {
                document.getElementById('error-box').classList.remove('hidden');
                document.getElementById('error-text').textContent = data.error;
            } else {
                document.getElementById('error-box').classList.add('hidden');
            }

            // Table
            const thead = document.getElementById('results-thead');
            const tbody = document.getElementById('results-tbody');
            thead.innerHTML = '';
            tbody.innerHTML = '';

            const rows = data.execution_rows || [];
            document.getElementById('row-count').textContent = rows.length + ' row' + (rows.length !== 1 ? 's' : '');

            if (rows.length === 0) {
                tbody.innerHTML = '<tr><td class="px-5 py-8 text-center text-slate-500" colspan="99">No rows returned</td></tr>';
                return;
            }

            const cols = Object.keys(rows[0]);
            const headerRow = document.createElement('tr');
            cols.forEach(c => {
                const th = document.createElement('th');
                th.className = 'px-5 py-3 text-left font-semibold';
                th.textContent = c;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);

            rows.forEach((row, i) => {
                const tr = document.createElement('tr');
                tr.className = 'hover:bg-slate-700/20 transition';
                if (i % 2 === 0) tr.classList.add('bg-slate-800/20');
                cols.forEach(c => {
                    const td = document.createElement('td');
                    td.className = 'px-5 py-2.5 text-slate-300 whitespace-nowrap';
                    td.textContent = row[c] ?? '';
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
        }

        // Enter key to submit
        document.getElementById('query-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') submitQuery();
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the interactive single-page dashboard."""
    return _DASHBOARD_HTML
