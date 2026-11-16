"""
AST-GuardCache Database Module.

Defines a 10-table multi-tenant EdTech/FinTech schema modeled after the Stride
production system.  Uses in-memory SQLite seeded with deterministic synthetic
data across two tenants so cross-tenant data leakage can be empirically tested.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

TABLE_DDLS: dict[str, str] = {
    "tenants": """
CREATE TABLE IF NOT EXISTS tenants (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    domain      TEXT    NOT NULL UNIQUE,
    plan        TEXT    NOT NULL DEFAULT 'free',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
    "students": """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   INTEGER NOT NULL REFERENCES tenants(id),
    name        TEXT    NOT NULL,
    email       TEXT    NOT NULL,
    department  TEXT    NOT NULL,
    cgpa        REAL    NOT NULL DEFAULT 0.0,
    year        INTEGER NOT NULL DEFAULT 1,
    status      TEXT    NOT NULL DEFAULT 'active',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
    "competency_tracks": """
CREATE TABLE IF NOT EXISTS competency_tracks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   INTEGER NOT NULL REFERENCES tenants(id),
    name        TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    difficulty  TEXT    NOT NULL DEFAULT 'intermediate',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
    "assessments": """
CREATE TABLE IF NOT EXISTS assessments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id        INTEGER NOT NULL REFERENCES tenants(id),
    student_id       INTEGER NOT NULL REFERENCES students(id),
    track_id         INTEGER NOT NULL REFERENCES competency_tracks(id),
    score            REAL    NOT NULL DEFAULT 0.0,
    max_score        REAL    NOT NULL DEFAULT 100.0,
    status           TEXT    NOT NULL DEFAULT 'pending',
    assessed_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
    "hackathons": """
CREATE TABLE IF NOT EXISTS hackathons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   INTEGER NOT NULL REFERENCES tenants(id),
    title       TEXT    NOT NULL,
    theme       TEXT    NOT NULL,
    start_date  TEXT    NOT NULL,
    end_date    TEXT    NOT NULL,
    prize_pool  REAL    NOT NULL DEFAULT 0.0,
    status      TEXT    NOT NULL DEFAULT 'upcoming'
);""",
    "hackathon_submissions": """
CREATE TABLE IF NOT EXISTS hackathon_submissions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id       INTEGER NOT NULL REFERENCES tenants(id),
    hackathon_id    INTEGER NOT NULL REFERENCES hackathons(id),
    student_id      INTEGER NOT NULL REFERENCES students(id),
    project_title   TEXT    NOT NULL,
    repo_url        TEXT,
    score           REAL    DEFAULT NULL,
    submitted_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
    "audit_logs": """
CREATE TABLE IF NOT EXISTS audit_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   INTEGER NOT NULL REFERENCES tenants(id),
    actor       TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    entity_type TEXT    NOT NULL,
    entity_id   INTEGER,
    details     TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
    "billing_invoices": """
CREATE TABLE IF NOT EXISTS billing_invoices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   INTEGER NOT NULL REFERENCES tenants(id),
    amount      REAL    NOT NULL,
    currency    TEXT    NOT NULL DEFAULT 'USD',
    status      TEXT    NOT NULL DEFAULT 'pending',
    due_date    TEXT    NOT NULL,
    paid_at     TEXT    DEFAULT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
    "webhook_events": """
CREATE TABLE IF NOT EXISTS webhook_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   INTEGER NOT NULL REFERENCES tenants(id),
    event_type  TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
    "notification_queue": """
CREATE TABLE IF NOT EXISTS notification_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   INTEGER NOT NULL REFERENCES tenants(id),
    recipient   TEXT    NOT NULL,
    channel     TEXT    NOT NULL DEFAULT 'email',
    subject     TEXT    NOT NULL,
    body        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'queued',
    sent_at     TEXT    DEFAULT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);""",
}

# ---------------------------------------------------------------------------
# Foreign-Key adjacency (1-hop closure map)
# ---------------------------------------------------------------------------

FK_GRAPH: dict[str, list[str]] = {
    "tenants": [],
    "students": ["tenants"],
    "competency_tracks": ["tenants"],
    "assessments": ["tenants", "students", "competency_tracks"],
    "hackathons": ["tenants"],
    "hackathon_submissions": ["tenants", "hackathons", "students"],
    "audit_logs": ["tenants"],
    "billing_invoices": ["tenants"],
    "webhook_events": ["tenants"],
    "notification_queue": ["tenants"],
}

# Columns per table (for schema-existence verification)
TABLE_COLUMNS: dict[str, list[str]] = {
    "tenants": ["id", "name", "domain", "plan", "created_at"],
    "students": [
        "id", "tenant_id", "name", "email", "department",
        "cgpa", "year", "status", "created_at",
    ],
    "competency_tracks": [
        "id", "tenant_id", "name", "category", "difficulty", "created_at",
    ],
    "assessments": [
        "id", "tenant_id", "student_id", "track_id",
        "score", "max_score", "status", "assessed_at",
    ],
    "hackathons": [
        "id", "tenant_id", "title", "theme",
        "start_date", "end_date", "prize_pool", "status",
    ],
    "hackathon_submissions": [
        "id", "tenant_id", "hackathon_id", "student_id",
        "project_title", "repo_url", "score", "submitted_at",
    ],
    "audit_logs": [
        "id", "tenant_id", "actor", "action",
        "entity_type", "entity_id", "details", "created_at",
    ],
    "billing_invoices": [
        "id", "tenant_id", "amount", "currency",
        "status", "due_date", "paid_at", "created_at",
    ],
    "webhook_events": [
        "id", "tenant_id", "event_type", "payload",
        "status", "attempts", "created_at",
    ],
    "notification_queue": [
        "id", "tenant_id", "recipient", "channel",
        "subject", "body", "status", "sent_at", "created_at",
    ],
}

# ---------------------------------------------------------------------------
# Opt-in wide schema (WIDE_SCHEMA=1): 50 distractor tables, for the retrieval
# experiments only. Off by default; the tables stay empty.
# ---------------------------------------------------------------------------

# table -> extra columns (after id, tenant_id); "x_id" REFERENCES x(id)
_WIDE_SPEC: dict[str, str] = {
    "departments": "name TEXT, code TEXT, head_name TEXT, budget REAL, created_at TEXT",
    "employees": "department_id INTEGER REFERENCES departments(id), full_name TEXT, work_email TEXT, job_title TEXT, hire_date TEXT, employment_type TEXT, status TEXT",
    "employee_contracts": "employee_id INTEGER REFERENCES employees(id), contract_type TEXT, start_date TEXT, end_date TEXT, signed_at TEXT",
    "payroll_runs": "period_start TEXT, period_end TEXT, status TEXT, total_gross REAL, total_net REAL, run_at TEXT",
    "payslips": "payroll_run_id INTEGER REFERENCES payroll_runs(id), employee_id INTEGER REFERENCES employees(id), gross_pay REAL, net_pay REAL, tax_withheld REAL, currency TEXT",
    "salary_bands": "title TEXT, min_salary REAL, max_salary REAL, currency TEXT, grade TEXT",
    "leave_requests": "employee_id INTEGER REFERENCES employees(id), leave_type TEXT, start_date TEXT, end_date TEXT, status TEXT, approved_by TEXT",
    "attendance_records": "employee_id INTEGER REFERENCES employees(id), work_date TEXT, check_in TEXT, check_out TEXT, hours REAL",
    "performance_reviews": "employee_id INTEGER REFERENCES employees(id), reviewer TEXT, period TEXT, rating INTEGER, comments TEXT",
    "job_postings": "department_id INTEGER REFERENCES departments(id), title TEXT, location TEXT, salary_min REAL, salary_max REAL, status TEXT, posted_at TEXT",
    "job_applicants": "job_posting_id INTEGER REFERENCES job_postings(id), full_name TEXT, email TEXT, stage TEXT, applied_at TEXT",
    "interviews": "applicant_id INTEGER REFERENCES job_applicants(id), interviewer TEXT, scheduled_at TEXT, outcome TEXT, notes TEXT",
    "warehouses": "name TEXT, city TEXT, country TEXT, capacity INTEGER, manager TEXT",
    "products": "sku TEXT, title TEXT, category TEXT, unit_price REAL, active INTEGER, created_at TEXT",
    "inventory_items": "warehouse_id INTEGER REFERENCES warehouses(id), product_id INTEGER REFERENCES products(id), quantity INTEGER, reorder_level INTEGER, updated_at TEXT",
    "suppliers": "company_name TEXT, contact_email TEXT, country TEXT, rating REAL, status TEXT",
    "purchase_orders": "supplier_id INTEGER REFERENCES suppliers(id), order_date TEXT, status TEXT, total_amount REAL, currency TEXT",
    "purchase_order_lines": "purchase_order_id INTEGER REFERENCES purchase_orders(id), product_id INTEGER REFERENCES products(id), quantity INTEGER, unit_cost REAL, received INTEGER",
    "stock_movements": "inventory_item_id INTEGER REFERENCES inventory_items(id), movement_type TEXT, quantity INTEGER, reason TEXT, moved_at TEXT",
    "shipments": "purchase_order_id INTEGER REFERENCES purchase_orders(id), carrier TEXT, tracking_code TEXT, status TEXT, shipped_at TEXT, delivered_at TEXT",
    "customers": "company_name TEXT, contact_name TEXT, email TEXT, phone TEXT, segment TEXT, created_at TEXT",
    "sales_orders": "customer_id INTEGER REFERENCES customers(id), order_date TEXT, status TEXT, total_amount REAL, currency TEXT",
    "sales_order_lines": "sales_order_id INTEGER REFERENCES sales_orders(id), product_id INTEGER REFERENCES products(id), quantity INTEGER, unit_price REAL, discount REAL",
    "quotes": "customer_id INTEGER REFERENCES customers(id), amount REAL, valid_until TEXT, status TEXT, created_at TEXT",
    "leads": "source TEXT, company_name TEXT, contact_email TEXT, score INTEGER, status TEXT, created_at TEXT",
    "opportunities": "lead_id INTEGER REFERENCES leads(id), stage TEXT, expected_value REAL, probability REAL, close_date TEXT",
    "marketing_campaigns": "name TEXT, channel TEXT, budget REAL, start_date TEXT, end_date TEXT, status TEXT",
    "campaign_responses": "campaign_id INTEGER REFERENCES marketing_campaigns(id), lead_id INTEGER REFERENCES leads(id), response_type TEXT, responded_at TEXT",
    "support_tickets": "customer_id INTEGER REFERENCES customers(id), subject TEXT, priority TEXT, status TEXT, opened_at TEXT, closed_at TEXT",
    "ticket_comments": "ticket_id INTEGER REFERENCES support_tickets(id), author TEXT, body TEXT, internal INTEGER, created_at TEXT",
    "sla_policies": "name TEXT, priority TEXT, response_hours INTEGER, resolution_hours INTEGER, active INTEGER",
    "knowledge_base_articles": "title TEXT, category TEXT, body TEXT, views INTEGER, published_at TEXT",
    "customer_feedback": "customer_id INTEGER REFERENCES customers(id), rating INTEGER, comment TEXT, channel TEXT, submitted_at TEXT",
    "expense_claims": "employee_id INTEGER REFERENCES employees(id), category TEXT, amount REAL, currency TEXT, status TEXT, submitted_at TEXT",
    "vendor_payments": "supplier_id INTEGER REFERENCES suppliers(id), amount REAL, currency TEXT, method TEXT, paid_at TEXT",
    "ledger_accounts": "code TEXT, name TEXT, account_type TEXT, balance REAL, currency TEXT",
    "journal_entries": "ledger_account_id INTEGER REFERENCES ledger_accounts(id), debit REAL, credit REAL, memo TEXT, posted_at TEXT",
    "tax_rates": "region TEXT, name TEXT, rate REAL, effective_from TEXT, active INTEGER",
    "budgets": "department_id INTEGER REFERENCES departments(id), fiscal_year INTEGER, planned_amount REAL, spent_amount REAL, status TEXT",
    "assets": "name TEXT, asset_tag TEXT, category TEXT, purchase_date TEXT, purchase_cost REAL, assigned_to TEXT",
    "maintenance_logs": "asset_id INTEGER REFERENCES assets(id), technician TEXT, description TEXT, cost REAL, performed_at TEXT",
    "projects": "department_id INTEGER REFERENCES departments(id), name TEXT, status TEXT, start_date TEXT, due_date TEXT, budget REAL",
    "project_tasks": "project_id INTEGER REFERENCES projects(id), title TEXT, assignee TEXT, status TEXT, priority TEXT, due_date TEXT",
    "timesheets": "employee_id INTEGER REFERENCES employees(id), project_id INTEGER REFERENCES projects(id), work_date TEXT, hours REAL, approved INTEGER",
    "meeting_rooms": "name TEXT, building TEXT, floor INTEGER, capacity INTEGER, has_projector INTEGER",
    "room_bookings": "room_id INTEGER REFERENCES meeting_rooms(id), booked_by TEXT, start_time TEXT, end_time TEXT, purpose TEXT",
    "api_keys": "label TEXT, key_prefix TEXT, scopes TEXT, created_at TEXT, revoked_at TEXT",
    "feature_flags": "flag_key TEXT, description TEXT, enabled INTEGER, rollout_percent INTEGER, updated_at TEXT",
    "user_sessions": "user_email TEXT, ip_address TEXT, user_agent TEXT, started_at TEXT, ended_at TEXT",
    "file_uploads": "uploaded_by TEXT, file_name TEXT, mime_type TEXT, size_bytes INTEGER, uploaded_at TEXT",
}


def _wide_tables() -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    """DDL, columns and FK neighbours of the distractor tables."""
    ddls, cols, fks = {}, {}, {}
    for table, spec in _WIDE_SPEC.items():
        parts = [c.strip() for c in spec.split(",")]
        body = ",\n    ".join(
            ["id          INTEGER PRIMARY KEY AUTOINCREMENT",
             "tenant_id   INTEGER NOT NULL REFERENCES tenants(id)", *parts]
        )
        ddls[table] = f"\nCREATE TABLE IF NOT EXISTS {table} (\n    {body}\n);"
        cols[table] = ["id", "tenant_id", *(p.split()[0] for p in parts)]
        fks[table] = ["tenants", *(p.split("REFERENCES ")[1].split("(")[0] for p in parts if "REFERENCES" in p)]
    return ddls, cols, fks


WIDE_DDLS, WIDE_COLUMNS, WIDE_FK_GRAPH = _wide_tables()

if os.getenv("WIDE_SCHEMA") == "1":
    TABLE_DDLS |= WIDE_DDLS
    TABLE_COLUMNS |= WIDE_COLUMNS
    FK_GRAPH |= WIDE_FK_GRAPH

# Tables that carry tenant_id (all except the tenants table itself)
TENANT_SCOPED_TABLES: set[str] = {
    t for t, cols in TABLE_COLUMNS.items() if "tenant_id" in cols
}

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_SQL = """
-- Tenants
INSERT INTO tenants (id, name, domain, plan) VALUES
    (1, 'Stride Academy',   'stride.edu',   'enterprise'),
    (2, 'Nova University',   'nova.edu',     'pro');

-- Students (Tenant 1)
INSERT INTO students (tenant_id, name, email, department, cgpa, year, status) VALUES
    (1, 'Alice Johnson',  'alice@stride.edu',  'CSE', 9.1, 3, 'active'),
    (1, 'Bob Smith',      'bob@stride.edu',    'CSE', 8.7, 3, 'active'),
    (1, 'Charlie Lee',    'charlie@stride.edu','ECE', 7.5, 2, 'active'),
    (1, 'Diana Patel',    'diana@stride.edu',  'ME',  8.2, 4, 'active'),
    (1, 'Ethan Brown',    'ethan@stride.edu',  'CSE', 6.9, 1, 'inactive');

-- Students (Tenant 2)
INSERT INTO students (tenant_id, name, email, department, cgpa, year, status) VALUES
    (2, 'Fatima Khan',    'fatima@nova.edu',   'CSE', 9.5, 4, 'active'),
    (2, 'George Wilson',  'george@nova.edu',   'ECE', 8.0, 2, 'active'),
    (2, 'Hannah Davis',   'hannah@nova.edu',   'ME',  7.8, 3, 'active');

-- Competency Tracks
INSERT INTO competency_tracks (tenant_id, name, category, difficulty) VALUES
    (1, 'Data Structures',    'core',     'intermediate'),
    (1, 'Machine Learning',   'elective', 'advanced'),
    (1, 'Web Development',    'core',     'beginner'),
    (2, 'Data Structures',    'core',     'intermediate'),
    (2, 'Cloud Computing',    'elective', 'advanced');

-- Assessments
INSERT INTO assessments (tenant_id, student_id, track_id, score, max_score, status) VALUES
    (1, 1, 1, 92.0, 100.0, 'completed'),
    (1, 1, 2, 88.5, 100.0, 'completed'),
    (1, 2, 1, 78.0, 100.0, 'completed'),
    (1, 3, 1, 65.0, 100.0, 'completed'),
    (1, 4, 3, 91.0, 100.0, 'completed'),
    (2, 6, 4, 95.0, 100.0, 'completed'),
    (2, 7, 4, 72.0, 100.0, 'completed');

-- Hackathons
INSERT INTO hackathons (tenant_id, title, theme, start_date, end_date, prize_pool, status) VALUES
    (1, 'CodeSprint 2025',   'AI for Good',      '2025-03-01', '2025-03-03', 5000.0,  'completed'),
    (1, 'HackNova 2025',     'FinTech',           '2025-06-15', '2025-06-17', 10000.0, 'upcoming'),
    (2, 'InnovateFest 2025', 'Sustainability',    '2025-04-10', '2025-04-12', 7500.0,  'completed');

-- Hackathon Submissions
INSERT INTO hackathon_submissions (tenant_id, hackathon_id, student_id, project_title, repo_url, score) VALUES
    (1, 1, 1, 'MediBot',        'https://github.com/alice/medibot',   88.0),
    (1, 1, 2, 'EduAssist',      'https://github.com/bob/eduassist',   76.0),
    (2, 3, 6, 'GreenTrack',     'https://github.com/fatima/greentrack', 92.0);

-- Audit Logs
INSERT INTO audit_logs (tenant_id, actor, action, entity_type, entity_id, details) VALUES
    (1, 'admin@stride.edu', 'CREATE', 'student', 1, 'Created student Alice Johnson'),
    (1, 'admin@stride.edu', 'UPDATE', 'student', 2, 'Updated CGPA for Bob Smith'),
    (2, 'admin@nova.edu',   'CREATE', 'student', 6, 'Created student Fatima Khan');

-- Billing Invoices
INSERT INTO billing_invoices (tenant_id, amount, currency, status, due_date) VALUES
    (1, 2500.00, 'USD', 'paid',    '2025-01-15'),
    (1, 2500.00, 'USD', 'pending', '2025-04-15'),
    (2, 1800.00, 'USD', 'paid',    '2025-02-01'),
    (2, 1800.00, 'USD', 'overdue', '2025-03-01');

-- Webhook Events
INSERT INTO webhook_events (tenant_id, event_type, payload, status, attempts) VALUES
    (1, 'student.created',    '{"student_id": 1}', 'delivered', 1),
    (1, 'assessment.graded',  '{"assessment_id": 1, "score": 92}', 'delivered', 1),
    (2, 'student.created',    '{"student_id": 6}', 'failed', 3);

-- Notification Queue
INSERT INTO notification_queue (tenant_id, recipient, channel, subject, body, status) VALUES
    (1, 'alice@stride.edu', 'email', 'Assessment Result', 'You scored 92 in Data Structures', 'sent'),
    (1, 'bob@stride.edu',   'email', 'Assessment Result', 'You scored 78 in Data Structures', 'queued'),
    (2, 'fatima@nova.edu',  'email', 'Welcome',           'Welcome to Nova University!',      'sent');
"""

# ---------------------------------------------------------------------------
# Database singleton (thread-safe, in-memory)
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()
_connection: sqlite3.Connection | None = None
_extra_sql = ""  # benchmark-only seed rows, see reset_connection()


def get_connection() -> sqlite3.Connection:
    """Return the shared in-memory SQLite connection, creating it on first call."""
    global _connection
    if _connection is not None:
        return _connection
    with _db_lock:
        if _connection is not None:
            return _connection
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        _init_schema(conn)
        # Defence in depth: the shared connection can never write after seeding
        conn.execute("PRAGMA query_only = ON;")
        _connection = conn
    return _connection


def reset_connection(extra_sql: str = "") -> sqlite3.Connection:
    """
    Tear down and recreate the database (useful for tests).

    *extra_sql* is run after the base seed; the benchmark uses it to add rows.
    The default database (and every test) uses the base seed only.
    """
    global _connection, _extra_sql
    with _db_lock:
        if _connection is not None:
            _connection.close()
        _connection = None
        _extra_sql = extra_sql
    return get_connection()


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and seed deterministic data."""
    cur = conn.cursor()
    for ddl in TABLE_DDLS.values():
        cur.executescript(ddl)
    cur.executescript(_SEED_SQL)
    if _extra_sql:
        cur.executescript(_extra_sql)
    conn.commit()


def execute_sql(sql: str, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """Execute a read-only SQL statement and return rows as list of dicts."""
    if conn is None:
        conn = get_connection()
    # One connection is shared across threads; serialise execute + fetch
    with _db_lock:
        cur = conn.cursor()
        cur.execute(sql)
        columns = unique_columns([desc[0] for desc in cur.description] if cur.description else [])
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def unique_columns(names: list[str]) -> list[str]:
    """
    Result column names made unique (name, name_2, ...). Two projected columns
    may share a name (``SELECT s.name, ct.name``); as dict keys the second would
    overwrite the first and a value would be lost before any comparison.
    """
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def norm_value(v: Any) -> Any:
    """
    Floats are compared to 12 significant digits: SUM/AVG over the same rows can
    differ in the last digits when the engine visits the rows in another order
    (e.g. 76.635 vs 76.63500000000002), which is not a different result.
    """
    return float(f"{v:.12g}") if isinstance(v, float) else v


def canon_rows(rows: list) -> list[str]:
    """Order-insensitive, float-tolerant form of a result set (rows as dicts or tuples)."""
    return sorted(
        repr(tuple(norm_value(v) for v in (r.values() if isinstance(r, dict) else r))) for r in rows
    )


def tenant_only_connection(tenant_id: int) -> sqlite3.Connection:
    """
    Copy of the database containing only *tenant_id*'s rows.

    Isolation oracle: a correctly scoped query must return the same result on
    the full database as the unscoped query returns on this copy.
    """
    src = get_connection()
    dst = sqlite3.connect(":memory:", check_same_thread=False)
    with _db_lock:
        src.backup(dst)
    dst.execute("PRAGMA query_only = OFF;")
    for table in TENANT_SCOPED_TABLES:
        dst.execute(f"DELETE FROM {table} WHERE tenant_id != ?", (tenant_id,))
    dst.execute("DELETE FROM tenants WHERE id != ?", (tenant_id,))
    dst.commit()
    dst.execute("PRAGMA query_only = ON;")
    dst.row_factory = sqlite3.Row
    return dst


def get_full_schema_text() -> str:
    """Return all DDLs concatenated (for zero-shot mode)."""
    return "\n".join(TABLE_DDLS.values())
