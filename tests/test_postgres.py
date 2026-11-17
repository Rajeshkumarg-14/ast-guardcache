"""
The scenario matrix on PostgreSQL, comparing three defences:

- verifier only:  verify_and_inject_ast(Q, t, "postgres") run as the owner (no RLS)
- RLS only:       Q run as app_reader with app.tenant_id = t
- both:           the verified query run under RLS

Oracle as in test_scenarios: every defence must return exactly the rows Q
returns on a schema holding only tenant t's data (iso_<t>).

Skipped unless the research database is up:
    docker compose -f docker-compose.research.yml up -d
"""

from __future__ import annotations

import psycopg
import pytest

from app import pg
from app.verifier import verify_and_inject_ast
from tests.test_scenarios import (
    COLUMNS,
    FKS,
    FUZZ,
    JOIN_SHAPES,
    KEY,
    PAIRS,
    SHAPES,
    TABLES,
    TENANTS,
    canon,
)

pytestmark = pytest.mark.skipif(not pg.available(), reason="research Postgres not running")


@pytest.fixture(scope="module", autouse=True)
def _db():
    pg.setup()


def _run(sql: str, **kw):
    """Rows, or the error class name if Postgres rejects the query."""
    try:
        return canon(pg.execute(sql, **kw))
    except psycopg.Error as exc:
        return type(exc).__name__


def assert_isolated_all(sql: str, tenant_id: int) -> None:
    """*sql* uses public.* where it is schema-qualified (iso run drops the qualifier)."""
    expected = _run(sql.replace("public.", ""), schema=f"iso_{tenant_id}")
    if isinstance(expected, str):
        pytest.skip(f"generated query is not executable on PostgreSQL ({expected})")
    assert _run(sql, rls_tenant=tenant_id) == expected, f"RLS only\nQ: {sql}"

    result = verify_and_inject_ast(sql, tenant_id, dialect="postgres")
    assert result.valid, result.error
    assert _run(result.sql) == expected, f"verifier only\nQ: {sql}\nscoped: {result.sql}"
    assert _run(result.sql, rls_tenant=tenant_id) == expected, f"both\nQ: {sql}"


# ---------------------------------------------------------------------------
# Benign queries: every defence must agree with the oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("shape", SHAPES)
def test_shape_isolation(shape, table, tenant_id):
    other = 2 if tenant_id == 1 else 1
    sql = shape.format(t=table, k=KEY[table], me=tenant_id, other=other)
    assert_isolated_all(sql.replace("main.", "public."), tenant_id)


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("fk", FKS, ids=lambda f: f"{f[0]}.{f[1]}->{f[2]}")
@pytest.mark.parametrize("shape", JOIN_SHAPES)
def test_fk_join_isolation(shape, fk, tenant_id):
    child, child_col, parent, parent_col = fk
    assert_isolated_all(shape.format(c=child, cc=child_col, p=parent, pc=parent_col), tenant_id)


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}x{p[1]}")
def test_table_pair_isolation(pair, tenant_id):
    a, b = pair
    assert_isolated_all(f"SELECT COUNT(*) FROM {a} AS a, {b} AS b", tenant_id)


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("col", COLUMNS, ids=lambda tc: f"{tc[0]}.{tc[1]}")
def test_every_column_isolated(col, tenant_id):
    table, column = col
    assert_isolated_all(f"SELECT {column} FROM {table}", tenant_id)


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("sql", FUZZ)
def test_fuzz_isolation(sql, tenant_id):
    assert_isolated_all(sql, tenant_id)


# ---------------------------------------------------------------------------
# Attacks: where RLS alone fails and the verifier is needed
# ---------------------------------------------------------------------------

# Rewrite the session variable RLS reads, inside the query itself
SPOOF = [
    "SELECT set_config('app.tenant_id','2',false) AS x, (SELECT COUNT(*) FROM students) AS n",
    "WITH s AS (SELECT set_config('app.tenant_id','2',false) AS x) SELECT s.x, st.name FROM s, students st",
    "SELECT name FROM students WHERE set_config('app.tenant_id','2',false) IS NOT NULL",
]
# Names that exist only in tenant 2 (see app/database.py seed data)
TENANT2_ONLY = {"Fatima Khan", "George Wilson", "Hannah Davis"}


def test_set_config_spoof_reads_foreign_rows_under_rls_only():
    """
    Session-variable RLS alone: a SELECT run for tenant 1 that overwrites
    app.tenant_id gets tenant 2's student rows back.
    """
    leaked = set()
    for sql in SPOOF[1:]:  # the two that project student names
        leaked |= {r["name"] for r in pg.execute(sql, rls_tenant=1)} & TENANT2_ONLY
    assert leaked, "expected tenant 2 names in the result of a tenant 1 session"


@pytest.mark.parametrize("sql", SPOOF)
def test_set_config_spoof_rejected_by_verifier(sql):
    result = verify_and_inject_ast(sql, 1, dialect="postgres")
    assert not result.valid and "set_config" in result.error


def test_explain_analyze_runs_under_rls_only():
    """EXPLAIN ANALYZE executes under RLS and returns plan text (row counts, filters)."""
    rows = pg.execute("EXPLAIN ANALYZE SELECT * FROM students WHERE cgpa > 100", rls_tenant=1)
    assert rows


PG_FORBIDDEN = [
    "EXPLAIN ANALYZE SELECT * FROM students WHERE cgpa > 100",
    "EXPLAIN SELECT * FROM students",
    "SELECT current_setting('app.tenant_id')",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_sleep(1)",
    "SELECT * FROM pg_catalog.pg_tables",
    "SELECT * FROM pg_tables",
    "SELECT * FROM pg_policies",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM iso_2.students",
    "SET app.tenant_id = '2'",
    "SET ROLE postgres",
    "RESET ROLE",
    "COPY students TO STDOUT",
    "LOCK TABLE students",
    "SELECT dblink('host=x', 'SELECT 1')",
    "SELECT * FROM students FOR UPDATE",
    "DO $$ BEGIN PERFORM 1; END $$",
    "WITH pg_x AS (SELECT 1 AS n) SELECT * FROM pg_x",
]


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("sql", PG_FORBIDDEN)
def test_pg_forbidden_rejected(sql, tenant_id):
    assert not verify_and_inject_ast(sql, tenant_id, dialect="postgres").valid
