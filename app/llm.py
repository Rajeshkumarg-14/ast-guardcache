"""
OpenRouter LLM Client with Self-Correction Retry Loop.

Wraps the OpenAI SDK configured for OpenRouter and implements a structured
retry loop: if the AST verifier rejects the generated SQL, the error message
is fed back to the LLM for up to ``max_retries`` additional attempts.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from app.config import settings

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


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
- Do NOT include any tenant_id filters — those are injected automatically.
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
                f"Question: {question}\n\n"
                "Write a single SQLite SELECT query."
            ),
        },
    ]

    last_error: str | None = None
    for attempt in range(1 + max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            raw_sql = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw_sql.startswith("```"):
                lines = raw_sql.split("\n")
                raw_sql = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            # If no verifier, return raw
            if verify_fn is None:
                return LLMResult(sql=raw_sql, retries_used=attempt)

            result = verify_fn(raw_sql, tenant_id)
            if result.valid:
                return LLMResult(sql=result.sql, retries_used=attempt)

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
            last_error = str(exc)
            if attempt == max_retries:
                break

    return LLMResult(
        sql="",
        retries_used=max_retries,
        error=f"Failed after {max_retries + 1} attempts. Last error: {last_error}",
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
        return LLMResult(sql=raw_sql, retries_used=0)

    result = verify_fn(raw_sql, tenant_id)
    if result.valid:
        return LLMResult(sql=result.sql, retries_used=0)

    return LLMResult(
        sql=raw_sql,
        retries_used=0,
        error=result.error,
    )
