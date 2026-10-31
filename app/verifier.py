"""
Symbolic AST Verifier & Tenant RLS Injector.

Uses sqlglot to parse candidate SQL and enforce three symbolic invariants:
1. Read-Only Guard        – the statement must be a query (SELECT / set operation)
                            with no DML/DDL anywhere in the tree and no unknown
                            or dangerous functions.
2. Schema Existence Guard – every table/column must exist in the metadata.
3. Tenant Scoping         – every reference to a tenant-scoped table, in every
                            scope (subqueries, CTEs, set operations, joins), is
                            replaced by ``(SELECT * FROM t WHERE tenant_id = ?) AS t``.
                            The ``tenants`` table is scoped on ``id``.

Scoping each table reference (instead of appending to the outer WHERE) keeps
outer-join semantics intact and means no scope can see another tenant's rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from app.database import TABLE_COLUMNS, TENANT_SCOPED_TABLES

# Statement types that must not appear anywhere in the tree
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Pragma, exp.Command, exp.Attach, exp.Detach,
)

# Functions sqlglot doesn't model natively are parsed as exp.Anonymous; only
# these are allowed (blocks load_extension, readfile, table-valued pragmas, ...)
_ALLOWED_ANONYMOUS_FUNCS = {"julianday", "printf", "instr", "total", "unixepoch"}

# Column each scoped table is filtered on
_SCOPE_KEY = {t: "tenant_id" for t in TENANT_SCOPED_TABLES} | {"tenants": "id"}

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Outcome of AST verification."""

    valid: bool
    sql: str  # The (possibly mutated) SQL
    error: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_and_inject_ast(
    sql: str, tenant_id: int
) -> VerificationResult:
    """
    Parse *sql*, apply the three symbolic guards, and scope every table to
    *tenant_id*.

    Returns a ``VerificationResult`` with the safe, tenant-scoped SQL or an
    explicit error message naming the offending element.
    """
    tenant_id = int(tenant_id)

    # --- Parse ----------------------------------------------------------------
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        return VerificationResult(valid=False, sql=sql, error=f"SQL parse error: {exc}")

    # --- 1. Read-Only Guard ---------------------------------------------------
    if not isinstance(parsed, exp.Query):
        stmt_type = type(parsed).__name__
        return VerificationResult(
            valid=False,
            sql=sql,
            error=f"Read-only violation: {stmt_type} statements are not allowed. Only SELECT queries are permitted.",
        )
    nested = next(parsed.find_all(*_FORBIDDEN_NODES), None)
    if nested is not None:
        return VerificationResult(
            valid=False,
            sql=sql,
            error=f"Read-only violation: nested {type(nested).__name__} is not allowed.",
        )
    for func in parsed.find_all(exp.Anonymous):
        if func.name.lower() not in _ALLOWED_ANONYMOUS_FUNCS:
            return VerificationResult(
                valid=False, sql=sql, error=f"Function not allowed: '{func.name}'."
            )

    # --- 2. Schema Existence Guard -------------------------------------------
    errors: list[str] = []
    cte_names = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)}
    # Sources whose columns we can't cheaply know (CTEs, derived tables)
    opaque_sources = cte_names | {
        sq.alias.lower() for sq in parsed.find_all(exp.Subquery) if sq.alias
    }
    referenced_tables: dict[str, str] = {}  # alias/name → real table name
    real_table_nodes: list[exp.Table] = []

    shadowing = cte_names & set(TABLE_COLUMNS)
    if shadowing:
        errors.append(f"CTE name shadows a real table: {sorted(shadowing)}.")

    for table_node in parsed.find_all(exp.Table):
        table_name = table_node.name.lower()
        if table_name in cte_names and not table_node.db:
            continue
        if table_node.db.lower() not in ("", "main") or table_node.catalog:
            errors.append(f"Schema-qualified table not allowed: '{table_node.sql(dialect='sqlite')}'.")
            continue
        if table_name not in TABLE_COLUMNS:
            errors.append(f"Hallucinated table: '{table_name or table_node.sql(dialect='sqlite')}' does not exist in the schema.")
            continue
        if table_name not in _SCOPE_KEY:
            errors.append(f"Table '{table_name}' has no tenant scope; refusing to query it.")
            continue
        referenced_tables[(table_node.alias or table_name).lower()] = table_name
        real_table_nodes.append(table_node)

    select_aliases = {a.alias.lower() for a in parsed.find_all(exp.Alias)}

    # Check columns
    for col_node in parsed.find_all(exp.Column):
        col_name = col_node.name.lower()
        tbl_ref = (col_node.table or "").lower()

        if tbl_ref in opaque_sources and tbl_ref not in referenced_tables:
            continue
        if tbl_ref and tbl_ref in referenced_tables:
            real_table = referenced_tables[tbl_ref]
            valid_cols = TABLE_COLUMNS.get(real_table, [])
            if col_name not in valid_cols:
                errors.append(
                    f"Hallucinated column: '{col_name}' does not exist in table '{real_table}'. "
                    f"Valid columns: {valid_cols}"
                )
        elif tbl_ref and tbl_ref in TABLE_COLUMNS:
            # Direct table reference without alias
            valid_cols = TABLE_COLUMNS[tbl_ref]
            if col_name not in valid_cols:
                errors.append(
                    f"Hallucinated column: '{col_name}' does not exist in table '{tbl_ref}'. "
                    f"Valid columns: {valid_cols}"
                )
        elif tbl_ref:
            errors.append(f"Unknown table reference: '{tbl_ref}' in column '{tbl_ref}.{col_name}'.")
        elif not opaque_sources and col_name not in select_aliases:
            # Unqualified column — must exist in some referenced table
            all_cols = set()
            for real in referenced_tables.values():
                all_cols.update(TABLE_COLUMNS.get(real, []))
            if referenced_tables and col_name not in all_cols:
                errors.append(
                    f"Hallucinated column: '{col_name}' not found in any referenced table."
                )

    if errors:
        return VerificationResult(
            valid=False,
            sql=sql,
            error=" | ".join(errors),
        )

    # --- 3. Tenant Scoping ----------------------------------------------------
    for table_node in real_table_nodes:
        real_table = table_node.name.lower()
        alias = table_node.args.get("alias") or exp.TableAlias(
            this=exp.to_identifier(real_table)
        )
        scoped = (
            exp.select("*")
            .from_(exp.Table(this=exp.to_identifier(real_table)))
            .where(
                exp.EQ(
                    this=exp.column(_SCOPE_KEY[real_table]),
                    expression=exp.Literal.number(tenant_id),
                )
            )
        )
        table_node.replace(exp.Subquery(this=scoped, alias=alias.copy()))

    mutated_sql = parsed.sql(dialect="sqlite")
    return VerificationResult(valid=True, sql=mutated_sql)
