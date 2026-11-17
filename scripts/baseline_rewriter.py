"""
Baseline A, "outer WHERE": the tenant-scoping step of the first version of our
verifier (git show a2624d3:app/verifier.py), kept here as a baseline only.

That version appended ``alias.tenant_id = t`` to the outermost WHERE clause for
every table alias found anywhere in the query. To compare the scoping steps
alone, the baseline first applies the current guards (read-only, catalog), so
both rewriters reject exactly the same statements, and then scopes the
original SQL the old way. Two choices favour the baseline: the ``tenants``
table is keyed on ``id`` as in the current verifier, and identifiers are built
as AST nodes (quoted where needed) rather than spliced as text. The old version
accepted only a plain SELECT at the root, so a set operation cannot be scoped
and counts as a rejection.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from app.database import TENANT_SCOPED_TABLES
from app.verifier import VerificationResult, _resolves_to_cte, verify_and_inject_ast

SET_OP_REJECTION = "baseline: only a plain SELECT can be scoped"


def baseline_scope(
    sql: str, tenant_id: int, dialect: str = "sqlite", schema: dict[str, list[str]] | None = None
) -> VerificationResult:
    guard = verify_and_inject_ast(sql, tenant_id, dialect=dialect, schema=schema)
    if not guard.valid:
        return guard
    parsed = sqlglot.parse_one(sql, read=dialect)
    if not isinstance(parsed, exp.Select):
        return VerificationResult(valid=False, sql=sql, error=SET_OP_REJECTION)
    if schema is None:
        key = {t: "tenant_id" for t in TENANT_SCOPED_TABLES} | {"tenants": "id"}
    else:
        key = {t: "tenant_id" for t, cols in schema.items() if "tenant_id" in cols}
    referenced: dict[str, str] = {}  # alias -> table, as in the old code (last one wins)
    for table in parsed.find_all(exp.Table):
        if _resolves_to_cte(table):
            continue
        referenced[table.alias_or_name] = table.name.lower()
    for alias, real in referenced.items():
        cond = exp.EQ(
            this=exp.Column(this=exp.to_identifier(key[real]), table=exp.to_identifier(alias)),
            expression=exp.Literal.number(int(tenant_id)),
        )
        parsed = parsed.where(cond, append=True)
    return VerificationResult(valid=True, sql=parsed.sql(dialect=dialect))


def features(sql: str, dialect: str = "sqlite") -> list[str]:
    """Query features used to classify failures."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return ["unparseable"]
    out = []
    setops = (exp.Union, exp.Except, exp.Intersect)
    if any(s is not tree and not isinstance(s.parent, (exp.CTE, *setops)) for s in tree.find_all(exp.Select)):
        out.append("subquery")
    if tree.find(exp.With):
        out.append("CTE")
    if tree.find(*setops):
        out.append("set operation")
    if any((j.side or "").upper() in ("LEFT", "RIGHT", "FULL") for j in tree.find_all(exp.Join)):
        out.append("outer join")
    names = [t.name.lower() for t in tree.find_all(exp.Table) if not _resolves_to_cte(t)]
    if len(names) != len(set(names)):
        out.append("table used twice")
    return out or ["single scope"]


if __name__ == "__main__":
    q = "SELECT * FROM students WHERE id IN (SELECT student_id FROM assessments)"
    r = baseline_scope(q, 1)
    # the subquery's table is filtered in the outer scope, where it is not visible
    assert r.valid and r.sql.endswith("students.tenant_id = 1) AND assessments.tenant_id = 1"), r.sql
    assert baseline_scope("SELECT id FROM students UNION SELECT id FROM tenants", 1).error == SET_OP_REJECTION
    assert features(q) == ["subquery"]
    print(r.sql)
