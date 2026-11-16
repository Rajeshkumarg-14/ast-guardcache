"""
OpenRouter LLM Client with Self-Correction Retry Loop.

Wraps the OpenAI SDK configured for OpenRouter and implements a structured
retry loop: if the AST verifier rejects the generated SQL, the error message
is fed back to the LLM for up to ``max_retries`` additional attempts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from app.config import settings

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: OpenAI | None = None
_last_call = 0.0

# OpenRouter caps ":free" models at 20 requests/minute
# ponytail: process-wide spacing, fine for the sequential benchmark; a shared
# limiter would be needed for concurrent live traffic on a free model.
_FREE_MIN_INTERVAL_S = 3.1

API_ERROR_PREFIX = "LLM API error"


def _throttle() -> None:
    global _last_call
    if settings.OPENROUTER_MODEL.endswith(":free"):
        time.sleep(max(0.0, _last_call + _FREE_MIN_INTERVAL_S - time.monotonic()))
    _last_call = time.monotonic()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return _client


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert SQLite SQL generator.  Given a user question and the relevant
table schemas, produce a SINGLE valid SQLite SELECT query.

Rules:
- Output ONLY the raw SQL query. No markdown, no explanations.
- Use only tables and columns present in the provided schema.
- Do NOT use aliases unless needed for self-joins.
- Do NOT add LIMIT unless the user explicitly asks for it.
- The user belongs to the tenant given below; only return that tenant's rows.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    """Result from the LLM text-to-SQL call."""

    sql: str
    retries_used: int
    error: str | None = None
    raw_sql: str = ""  # LLM output before AST tenant injection (cache input)
    finish_reason: str = ""  # of the last completion ("length" = output limit hit)
    verify_ms: float = 0.0  # time spent in the AST verifier across attempts
    prompt_tokens: int = 0  # summed over attempts, as reported by the endpoint


def generate_sql(
    question: str,
    schema_ddl: str,
    max_retries: int | None = None,
    verify_fn: callable = None,
    tenant_id: int = 1,
) -> LLMResult:
    """
    Generate SQL from natural language via OpenRouter, with optional
    AST-based self-correction.

    Parameters
    ----------
    question : str
        Natural-language user question.
    schema_ddl : str
        Concatenated DDL strings for the relevant tables.
    max_retries : int, optional
        Maximum correction attempts (default from settings).
    verify_fn : callable, optional
        ``verify_and_inject_ast(sql, tenant_id)`` callable for the retry loop.
    tenant_id : int
        Current tenant for RLS injection.
    """
    if max_retries is None:
        max_retries = settings.MAX_AST_RETRIES

    client = _get_client()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema_ddl}\n\n"
                f"Current tenant_id: {tenant_id}\n"
                f"Question: {question}\n\n"
                "Write a single SQLite SELECT query."
            ),
        },
    ]

    last_error: str | None = None
    finish = ""
    verify_ms = 0.0
    prompt_tokens = 0
    for attempt in range(1 + max_retries):
        try:
            _throttle()
            response = client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,  # room for hidden reasoning tokens on reasoning models
            )
            raw_sql = (response.choices[0].message.content or "").strip()
            finish = response.choices[0].finish_reason or ""
            prompt_tokens += getattr(response.usage, "prompt_tokens", 0) or 0

            # Strip markdown code fences if present
            if raw_sql.startswith("```"):
                lines = raw_sql.split("\n")
                raw_sql = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            # If no verifier, return raw
            if verify_fn is None:
                return LLMResult(
                    sql=raw_sql, retries_used=attempt, raw_sql=raw_sql,
                    finish_reason=finish, prompt_tokens=prompt_tokens,
                )

            t0 = time.perf_counter()
            result = verify_fn(raw_sql, tenant_id)
            verify_ms += (time.perf_counter() - t0) * 1000
            if result.valid:
                return LLMResult(
                    sql=result.sql, retries_used=attempt, raw_sql=raw_sql,
                    finish_reason=finish, verify_ms=verify_ms, prompt_tokens=prompt_tokens,
                )

            # Feed error back for correction
            last_error = result.error
            messages.append({"role": "assistant", "content": raw_sql})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The SQL you generated has an error:\n{last_error}\n\n"
                        "Please fix the query and output ONLY the corrected SQL."
                    ),
                }
            )

        except Exception as exc:
            # Provider failures (no credits, rate limit, network) are not model
            # errors: surface them instead of scoring them as rejected SQL.
            if attempt == max_retries:
                raise RuntimeError(f"{API_ERROR_PREFIX}: {exc}") from exc

    return LLMResult(
        sql="",
        retries_used=max_retries,
        error=f"Failed after {max_retries + 1} attempts. Last error: {last_error}",
        finish_reason=finish,
        verify_ms=verify_ms,
        prompt_tokens=prompt_tokens,
    )


# ---------------------------------------------------------------------------
# Mock LLM for offline / deterministic testing
# ---------------------------------------------------------------------------

# Mapping of question keywords → deterministic SQL responses
_MOCK_RESPONSES: dict[str, str] = {
    "students in cse": "SELECT * FROM students WHERE department = 'CSE'",
    "students in ece": "SELECT * FROM students WHERE department = 'ECE'",
    "cgpa > 8": "SELECT * FROM students WHERE cgpa > 8.0",
    "cgpa > 9": "SELECT * FROM students WHERE cgpa > 9.0",
    "total students": "SELECT COUNT(*) AS total_students FROM students",
    "average cgpa": "SELECT AVG(cgpa) AS avg_cgpa FROM students",
    "hackathon": "SELECT h.title, hs.project_title, hs.score FROM hackathons h JOIN hackathon_submissions hs ON h.id = hs.hackathon_id",
    "assessment": "SELECT s.name, a.score, ct.name AS track FROM assessments a JOIN students s ON a.student_id = s.id JOIN competency_tracks ct ON a.track_id = ct.id",
    "billing": "SELECT * FROM billing_invoices WHERE status = 'pending'",
    "invoice": "SELECT * FROM billing_invoices",
    "overdue": "SELECT * FROM billing_invoices WHERE status = 'overdue'",
    "notification": "SELECT * FROM notification_queue WHERE status = 'queued'",
    "webhook": "SELECT * FROM webhook_events WHERE status = 'failed'",
    "audit": "SELECT * FROM audit_logs ORDER BY created_at DESC",
    "competency": "SELECT * FROM competency_tracks",
    "active students": "SELECT * FROM students WHERE status = 'active'",
    "departments": "SELECT DISTINCT department FROM students",
    "top scorer": "SELECT s.name, a.score FROM assessments a JOIN students s ON a.student_id = s.id ORDER BY a.score DESC LIMIT 5",
}


def mock_generate_sql(
    question: str,
    schema_ddl: str,
    verify_fn: callable = None,
    tenant_id: int = 1,
) -> LLMResult:
    """
    Deterministic mock LLM for offline benchmarking and tests.

    Matches question keywords to pre-defined SQL responses.
    """
    q_lower = question.lower()

    # Find best matching mock
    best_key = None
    best_match_len = 0
    for key in _MOCK_RESPONSES:
        if key in q_lower and len(key) > best_match_len:
            best_key = key
            best_match_len = len(key)

    if best_key is None:
        # Default fallback
        raw_sql = "SELECT * FROM students"
    else:
        raw_sql = _MOCK_RESPONSES[best_key]

    if verify_fn is None:
        return LLMResult(sql=raw_sql, retries_used=0, raw_sql=raw_sql)

    t0 = time.perf_counter()
    result = verify_fn(raw_sql, tenant_id)
    verify_ms = (time.perf_counter() - t0) * 1000
    if result.valid:
        return LLMResult(sql=result.sql, retries_used=0, raw_sql=raw_sql, verify_ms=verify_ms)

    return LLMResult(
        sql=raw_sql,
        retries_used=0,
        error=result.error,
    )
