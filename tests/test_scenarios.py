"""
Generated scenario matrix (1000+ cases) for the paper's three claims.

Isolation oracle used throughout: for any query Q and tenant t,
    rows(verify(Q, t) on the full database) == rows(Q on a database holding only t's rows)
Equality proves both that no other tenant's data reaches the result and that
scoping does not change the query's meaning for the caller.
"""

from __future__ import annotations

import random
import sqlite3

import pytest

from app.cache import ParameterizedSemanticCache
from app.database import (
    TABLE_COLUMNS,
    execute_sql,
    get_connection,
    tenant_only_connection,
)
from app.pipeline import Pipeline
from app.verifier import verify_and_inject_ast

TENANTS = (1, 2)
TABLES = sorted(TABLE_COLUMNS)
KEY = {t: ("id" if t == "tenants" else "tenant_id") for t in TABLES}

_iso: dict[int, sqlite3.Connection] = {}


def iso(tenant_id: int) -> sqlite3.Connection:
    if tenant_id not in _iso:
        _iso[tenant_id] = tenant_only_connection(tenant_id)
    return _iso[tenant_id]


def canon(rows: list[dict]) -> list[str]:
    return sorted(repr(tuple(r.values())) for r in rows)


def _run(sql: str, conn=None):
    """Rows, or the error class name if SQLite rejects the query."""
    try:
        return canon(execute_sql(sql, conn=conn))
    except sqlite3.Error as exc:
        return type(exc).__name__


def assert_isolated(sql: str, tenant_id: int) -> None:
    result = verify_and_inject_ast(sql, tenant_id)
    assert result.valid, result.error
    scoped = _run(result.sql)
    expected = _run(sql, conn=iso(tenant_id))
    assert scoped == expected, f"\nQ:      {sql}\nscoped: {result.sql}"


def _foreign_keys() -> list[tuple[str, str, str, str]]:
    """(child, child_col, parent, parent_col) from SQLite's own FK metadata."""
    conn = get_connection()
    out = []
    for child in TABLES:
        for row in conn.execute(f"PRAGMA foreign_key_list({child})").fetchall():
            out.append((child, row["from"], row["table"], row["to"] or "id"))
    return out


FKS = _foreign_keys()

# ---------------------------------------------------------------------------
# 1. Query shapes × tables × tenants
# ---------------------------------------------------------------------------

SHAPES = [
    "SELECT * FROM {t}",
    "SELECT COUNT(*) FROM {t}",
    "SELECT * FROM {t} AS x",
    "SELECT (SELECT COUNT(*) FROM {t}) AS c",
    "SELECT * FROM {t} WHERE id IN (SELECT id FROM {t})",
    "SELECT * FROM (SELECT * FROM {t}) AS d",
    "WITH c AS (SELECT * FROM {t}) SELECT * FROM c",
    "WITH c AS (SELECT * FROM {t}) SELECT COUNT(*) FROM c AS c1, c AS c2",
    "SELECT id FROM {t} UNION SELECT id FROM {t}",
    "SELECT id FROM {t} UNION ALL SELECT id FROM {t}",
    "SELECT id FROM {t} EXCEPT SELECT id FROM {t} WHERE id > 2",
    "SELECT id FROM {t} INTERSECT SELECT id FROM {t}",
    "SELECT * FROM {t} WHERE EXISTS (SELECT 1 FROM {t} AS y WHERE y.id > 0)",
    "SELECT * FROM {t} WHERE NOT EXISTS (SELECT 1 FROM {t} AS y WHERE y.id < 0)",
    "SELECT COUNT(*) FROM {t} AS a, {t} AS b",
    "SELECT * FROM {t} WHERE 1 = 1 OR 1 = 1",
    "SELECT * FROM {t} WHERE {k} = {other}",
    "SELECT * FROM {t} WHERE {k} != {me}",
    "SELECT * FROM {t} WHERE {k} IN (1, 2)",
    "SELECT * FROM {t} WHERE {k} = {me} OR {k} = {other}",
    "SELECT MAX(id), MIN(id), SUM(id), AVG(id) FROM {t}",
    "SELECT * FROM main.{t}",
    "SELECT * FROM {t} LIMIT 100 OFFSET 0",
    "SELECT * FROM {t} ORDER BY id DESC",
    "SELECT * FROM {t} WHERE id = (SELECT MAX(id) FROM {t})",
    "SELECT * FROM {t} AS t1 LEFT JOIN {t} AS t2 ON t1.id = t2.id + 1",
    "SELECT {k}, COUNT(*) AS n FROM {t} GROUP BY {k} HAVING COUNT(*) > 0",
    "SELECT COUNT(*) AS n FROM {t} ORDER BY n",
    "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM {t}",
    "SELECT * FROM {t} WHERE id > (SELECT AVG(id) FROM {t})",
    "SELECT CASE WHEN (SELECT COUNT(*) FROM {t}) > 3 THEN 'many' ELSE 'few' END AS bucket",
    "SELECT * FROM (SELECT * FROM (SELECT * FROM {t}) AS a) AS b",
]


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("shape", SHAPES)
def test_shape_isolation(shape, table, tenant_id):
    other = 2 if tenant_id == 1 else 1
    sql = shape.format(t=table, k=KEY[table], me=tenant_id, other=other)
    assert_isolated(sql, tenant_id)


# ---------------------------------------------------------------------------
# 2. Foreign-key joins (inner, left, subquery, aggregate) × tenants
# ---------------------------------------------------------------------------

JOIN_SHAPES = [
    "SELECT * FROM {c} AS c JOIN {p} AS p ON c.{cc} = p.{pc}",
    "SELECT * FROM {p} AS p LEFT JOIN {c} AS c ON c.{cc} = p.{pc}",
    "SELECT p.id, COUNT(c.id) AS n FROM {p} AS p LEFT JOIN {c} AS c ON c.{cc} = p.{pc} GROUP BY p.id",
    "SELECT * FROM {p} WHERE {pc} IN (SELECT {cc} FROM {c})",
    "SELECT * FROM {c} WHERE {cc} NOT IN (SELECT {pc} FROM {p})",
    "SELECT COUNT(*) FROM {c} AS c JOIN {p} AS p ON c.{cc} = p.{pc} WHERE p.{pc} > 0",
]


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("fk", FKS, ids=lambda f: f"{f[0]}.{f[1]}->{f[2]}")
@pytest.mark.parametrize("shape", JOIN_SHAPES)
def test_fk_join_isolation(shape, fk, tenant_id):
    child, child_col, parent, parent_col = fk
    assert_isolated(shape.format(c=child, cc=child_col, p=parent, pc=parent_col), tenant_id)


# ---------------------------------------------------------------------------
# 3. Every pair of tables (cross join and correlated EXISTS) × tenants
# ---------------------------------------------------------------------------

PAIRS = [(a, b) for a in TABLES for b in TABLES]


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}x{p[1]}")
def test_table_pair_isolation(pair, tenant_id):
    a, b = pair
    assert_isolated(f"SELECT COUNT(*) FROM {a} AS a, {b} AS b", tenant_id)
    assert_isolated(
        f"SELECT COUNT(*) FROM {a} AS a WHERE EXISTS (SELECT 1 FROM {b} AS b WHERE b.id >= a.id)",
        tenant_id,
    )


# ---------------------------------------------------------------------------
# 4. Every real column, qualified and unqualified × tenants
# ---------------------------------------------------------------------------

COLUMNS = [(t, c) for t in TABLES for c in TABLE_COLUMNS[t]]


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("col", COLUMNS, ids=lambda tc: f"{tc[0]}.{tc[1]}")
def test_every_column_valid_and_isolated(col, tenant_id):
    table, column = col
    assert_isolated(f"SELECT {column} FROM {table}", tenant_id)
    assert_isolated(f"SELECT x.{column} FROM {table} AS x ORDER BY x.{column}", tenant_id)


# ---------------------------------------------------------------------------
# 5. Seeded random fuzz: nested compositions of shapes
# ---------------------------------------------------------------------------


def _random_query(rng: random.Random, depth: int = 0) -> str:
    table = rng.choice(TABLES)
    if depth >= 2 or rng.random() < 0.35:
        return f"SELECT * FROM {table}"
    inner = _random_query(rng, depth + 1)
    other = rng.choice(TABLES)
    return rng.choice([
        f"SELECT * FROM ({inner}) AS d{depth}",
        f"SELECT COUNT(*) AS n FROM ({inner}) AS d{depth}",
        f"SELECT * FROM {other} WHERE id IN (SELECT id FROM ({inner}) AS d{depth})",
        f"WITH w{depth} AS ({inner}) SELECT COUNT(*) AS n FROM w{depth}",
        f"SELECT (SELECT COUNT(*) FROM ({inner}) AS d{depth}) AS n FROM {other}",
        f"SELECT id FROM {other} UNION SELECT id FROM ({inner}) AS d{depth}",
        f"SELECT * FROM {other} AS o LEFT JOIN ({inner}) AS d{depth} ON o.id = d{depth}.id",
    ])


FUZZ = [_random_query(random.Random(seed)) for seed in range(300)]


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("sql", FUZZ)
def test_fuzz_isolation(sql, tenant_id):
    assert_isolated(sql, tenant_id)


# ---------------------------------------------------------------------------
# 6. Forbidden statements and escape hatches are rejected
# ---------------------------------------------------------------------------

FORBIDDEN = [
    "DROP TABLE students",
    "DELETE FROM students",
    "UPDATE students SET cgpa = 10",
    "UPDATE students SET tenant_id = 1",
    "INSERT INTO students (name) VALUES ('x')",
    "INSERT INTO students SELECT * FROM students",
    "REPLACE INTO students (id) VALUES (1)",
    "ALTER TABLE students ADD COLUMN x INT",
    "CREATE TABLE x (a INT)",
    "CREATE INDEX i ON students (name)",
    "CREATE VIEW v AS SELECT * FROM students",
    "CREATE TEMP TABLE t AS SELECT * FROM students",
    "PRAGMA query_only = OFF",
    "PRAGMA table_info(students)",
    "ATTACH DATABASE 'x.db' AS x",
    "DETACH DATABASE x",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
    "BEGIN",
    "COMMIT",
    "SAVEPOINT a",
    "SELECT 1; DROP TABLE students",
    "SELECT * FROM students; DELETE FROM students",
    "WITH x AS (SELECT 1) DELETE FROM students",
    "SELECT load_extension('evil')",
    "SELECT readfile('/etc/passwd')",
    "SELECT writefile('x', 'y')",
    "SELECT fts3_tokenizer('simple')",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM sqlite_schema",
    "SELECT * FROM sqlite_temp_master",
    "SELECT * FROM pragma_table_info('students')",
    "SELECT * FROM temp.students",
    "SELECT * FROM aux.students",
    "WITH students AS (SELECT * FROM students) SELECT * FROM students",
    "WITH tenants AS (SELECT 1 AS id) SELECT * FROM tenants",
    # CTE defined in an inner scope must not hide a same-named table outside it
    "SELECT * FROM (WITH sqlite_master AS (SELECT 1 AS n) SELECT * FROM sqlite_master) AS x, sqlite_master",
    "SELECT * FROM (WITH c AS (SELECT 1 AS n) SELECT * FROM c) AS x, c",
    "SELECT * FROM (WITH sqlite_schema AS (SELECT 1 AS n) SELECT * FROM sqlite_schema) AS x",
    "SELECT (SELECT COUNT(*) FROM (WITH pragma_x AS (SELECT 1) SELECT * FROM pragma_x)) AS n",
    "SELECT * FROM students AS s WHERE s.id IN (SELECT id FROM secret_table)",
    "SELECT nope.name FROM students",
    "THIS IS NOT SQL",
]


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("sql", FORBIDDEN)
def test_forbidden_rejected(sql, tenant_id):
    assert not verify_and_inject_ast(sql, tenant_id).valid


WRITES = [s for s in FORBIDDEN[:12]]


@pytest.mark.parametrize("sql", WRITES)
def test_database_refuses_writes_even_if_verifier_bypassed(sql):
    """Defence in depth: the shared connection is query_only."""
    with pytest.raises(sqlite3.Error):
        execute_sql(sql)


# ---------------------------------------------------------------------------
# 7. Hallucinated identifiers are rejected
# ---------------------------------------------------------------------------

FAKE_COLUMNS = ["full_name", "student_name", "gpa", "salary", "password", "tenantid", "ssn", "email_address"]
FAKE_TABLES = ["users", "student", "accounts", "payments", "invoices", "teams", "grades", "admins", "logs", "secrets"]


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("column", FAKE_COLUMNS)
def test_hallucinated_column_rejected(table, column):
    for sql in (f"SELECT x.{column} FROM {table} AS x", f"SELECT {table}.{column} FROM {table}"):
        result = verify_and_inject_ast(sql, 1)
        assert not result.valid and "Hallucinated column" in result.error


@pytest.mark.parametrize("table", FAKE_TABLES)
def test_hallucinated_table_rejected(table):
    result = verify_and_inject_ast(f"SELECT * FROM {table}", 1)
    assert not result.valid and "Hallucinated table" in result.error


# ---------------------------------------------------------------------------
# 8. Cache: same operator hits and rebinds; different operator never hits
# ---------------------------------------------------------------------------

FIELDS = [
    ("students", "students", "cgpa", "8.5", "9.0"),
    ("assessments", "assessments", "score", "80", "90"),
    ("billing_invoices", "billing invoices", "amount", "1000", "2000"),
    ("students", "students", "year", "2", "3"),
]
PHRASES = {
    "gt": ([">", "above", "more than", "greater than", "over"], ">"),
    "lt": (["<", "below", "less than", "under", "fewer than"], "<"),
    "ge": ([">=", "at least", "no less than"], ">="),
    "le": (["<=", "at most", "no more than"], "<="),
}
PHRASE_LIST = [(cls, p) for cls, (ps, _) in PHRASES.items() for p in ps]


def _q(words, field, phrase, value):
    return f"Show {words} with {field} {phrase} {value}"


def _sql(table, field, cls, value):
    return f"SELECT * FROM {table} WHERE {field} {PHRASES[cls][1]} {value}"


@pytest.fixture
def cache():
    return ParameterizedSemanticCache()


@pytest.mark.parametrize("field", FIELDS, ids=lambda f: f[2])
@pytest.mark.parametrize("phrase", PHRASE_LIST, ids=lambda p: f"{p[0]}:{p[1]}")
def test_cache_same_phrase_hits_and_rebinds(cache, field, phrase):
    table, words, col, v1, v2 = field
    cls, p = phrase
    cache.store(_q(words, col, p, v1), _sql(table, col, cls, v1), tenant_id=1)
    sql, hit = cache.lookup(_q(words, col, p, v2), tenant_id=1)
    assert hit
    assert f"{col} {PHRASES[cls][1]} {v2}" in sql
    assert_isolated(sql, 1)


SAME_CLASS = [(a, b) for a in PHRASE_LIST for b in PHRASE_LIST if a[0] == b[0] and a != b]


@pytest.mark.parametrize("field", FIELDS, ids=lambda f: f[2])
@pytest.mark.parametrize("pair", SAME_CLASS, ids=lambda ab: f"{ab[0][1]}->{ab[1][1]}")
def test_cache_paraphrase_never_changes_operator(cache, field, pair):
    """A hit across paraphrases is optional; a wrong operator is never allowed."""
    table, words, col, v1, v2 = field
    (cls, p1), (_, p2) = pair
    cache.store(_q(words, col, p1, v1), _sql(table, col, cls, v1), tenant_id=1)
    sql, hit = cache.lookup(_q(words, col, p2, v2), tenant_id=1)
    if hit:
        assert f"{col} {PHRASES[cls][1]} {v2}" in sql


CROSS_CLASS = [(a, b) for a in PHRASE_LIST for b in PHRASE_LIST if a[0] != b[0]]


@pytest.mark.parametrize("field", FIELDS, ids=lambda f: f[2])
@pytest.mark.parametrize("pair", CROSS_CLASS, ids=lambda ab: f"{ab[0][1]}->{ab[1][1]}")
def test_cache_near_miss_operator_never_hits(cache, field, pair):
    table, words, col, v1, v2 = field
    (cls, p1), (_, p2) = pair
    cache.store(_q(words, col, p1, v1), _sql(table, col, cls, v1), tenant_id=1)
    _, hit = cache.lookup(_q(words, col, p2, v2), tenant_id=1)
    assert not hit


DIRECTION = [
    ("top", "DESC"), ("highest", "DESC"), ("best", "DESC"),
    ("bottom", "ASC"), ("lowest", "ASC"), ("worst", "ASC"),
]


@pytest.mark.parametrize("a", DIRECTION, ids=lambda d: d[0])
@pytest.mark.parametrize("b", DIRECTION, ids=lambda d: d[0])
def test_cache_sort_direction(cache, a, b):
    cache.store(f"Show {a[0]} 5 students by cgpa",
                f"SELECT * FROM students ORDER BY cgpa {a[1]} LIMIT 5", tenant_id=1)
    sql, hit = cache.lookup(f"Show {b[0]} 3 students by cgpa", tenant_id=1)
    if a[1] != b[1]:
        assert not hit
    elif hit:
        assert f"ORDER BY cgpa {a[1]} LIMIT 3" in sql


NEGATION = [
    ("Show students in 'CSE'", "Show students not in 'ECE'"),
    ("Show invoices with status 'paid'", "Show invoices without status 'pending'"),
    ("List students in 'ME'", "List students except 'ECE'"),
    ("Show tracks in 'backend'", "Show tracks excluding 'frontend'"),
]


@pytest.mark.parametrize("pair", NEGATION)
@pytest.mark.parametrize("order", ["pos_first", "neg_first"])
def test_cache_negation_never_hits(cache, pair, order):
    first, second = pair if order == "pos_first" else pair[::-1]
    cache.store(first, "SELECT * FROM students WHERE department = 'CSE'", tenant_id=1)
    _, hit = cache.lookup(second, tenant_id=1)
    assert not hit


DEPTS = ["CSE", "ECE", "ME", "CIVIL", "EEE"]


@pytest.mark.parametrize("a", DEPTS)
@pytest.mark.parametrize("b", DEPTS)
def test_cache_unquoted_entity_never_reused(cache, a, b):
    cache.store(f"Show students in {a} with cgpa > 8",
                f"SELECT * FROM students WHERE department = '{a}' AND cgpa > 8", tenant_id=1)
    sql, hit = cache.lookup(f"Show students in {b} with cgpa > 7", tenant_id=1)
    if a != b:
        assert not hit or f"'{a}'" not in sql


PAYLOADS = [
    "x' OR '1'='1", "'; DROP TABLE students; --", "CSE' UNION SELECT * FROM tenants --",
    "1 OR 1=1", "%' OR tenant_id = 2 --", "\\'", "''", "CSE\" OR \"1\"=\"1",
    "') OR ('a'='a", "x'); DELETE FROM students; --", "admin'--", "' OR tenant_id != 1 --",
    "CSE' AND (SELECT COUNT(*) FROM tenants) > 0 --", "Robert'); DROP TABLE students;--",
    "' UNION ALL SELECT name, NULL FROM tenants --", "%", "_", "\x00", "CSE'/*", "*/ OR 1=1 --",
]


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("payload", PAYLOADS)
def test_cache_rebinding_is_injection_safe(cache, payload, tenant_id):
    cache.store("Show students in department \"CSE\"",
                "SELECT * FROM students WHERE department = 'CSE'", tenant_id=tenant_id)
    sql, hit = cache.lookup(f"Show students in department \"{payload}\"", tenant_id=tenant_id)
    if hit:
        result = verify_and_inject_ast(sql, tenant_id)
        assert result.valid, result.error
        rows = execute_sql(result.sql)
        assert all(r.get("tenant_id") == tenant_id for r in rows)
        assert_isolated(sql, tenant_id)


@pytest.mark.parametrize("owner", TENANTS)
@pytest.mark.parametrize("field", FIELDS, ids=lambda f: f[2])
def test_cache_is_partitioned_by_tenant(cache, owner, field):
    table, words, col, v1, v2 = field
    cache.store(_q(words, col, ">", v1), _sql(table, col, "gt", v1), tenant_id=owner)
    _, hit = cache.lookup(_q(words, col, ">", v2), tenant_id=3 - owner)
    assert not hit


# ---------------------------------------------------------------------------
# 9. End-to-end pipeline (mock LLM): verified modes never leak; oracle has power
# ---------------------------------------------------------------------------

MOCK_QUESTIONS = [
    "Show students in cse", "Show students in ece", "students with cgpa > 8", "cgpa > 9 students",
    "total students", "average cgpa", "hackathon results", "assessment report", "billing summary",
    "all invoices", "overdue payments", "notification backlog", "webhook failures", "audit trail",
    "competency tracks", "active students", "list departments", "top scorer", "something unknown",
]


@pytest.fixture(scope="module")
def mock_pipeline():
    return Pipeline(use_mock_llm=True)


def _leaks(sql: str, tenant_id: int) -> bool:
    return canon(execute_sql(sql)) != canon(execute_sql(sql, conn=iso(tenant_id)))


@pytest.mark.parametrize("tenant_id", TENANTS)
@pytest.mark.parametrize("mode", ["rag_ast", "ast_guardcache"])
@pytest.mark.parametrize("question", MOCK_QUESTIONS)
def test_pipeline_verified_modes_never_leak(mock_pipeline, question, mode, tenant_id):
    result = mock_pipeline.run(question=question, tenant_id=tenant_id, mode=mode)
    assert result.error is None, result.error
    assert not _leaks(result.verified_sql, tenant_id)


@pytest.mark.parametrize("tenant_id", TENANTS)
def test_oracle_detects_unverified_leaks(mock_pipeline, tenant_id):
    """Sanity: the same oracle flags leaks in the unverified baseline."""
    leaks = 0
    for q in MOCK_QUESTIONS:
        result = mock_pipeline.run(question=q, tenant_id=tenant_id, mode="zero_shot")
        if result.error is None and _leaks(result.verified_sql, tenant_id):
            leaks += 1
    assert leaks > 0
