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
    exp.Copy, exp.Set, exp.Lock,
)

# Schema qualifiers that refer to the application tables, per dialect
_DEFAULT_SCHEMAS = {"sqlite": ("", "main"), "postgres": ("", "public")}

# Functions sqlglot doesn't model natively are parsed as exp.Anonymous; only
# these are allowed (blocks load_extension, readfile, table-valued pragmas, ...)
_ALLOWED_ANONYMOUS_FUNCS = {
    "julianday", "printf", "instr", "total", "unixepoch", "datetime", "date", "time",
}

# Functions sqlglot does model but that reveal the server environment rather
# than tenant data (current_user, version(), current_database(), ...)
_ENVIRONMENT_NODES = tuple(
    cls for cls in (
        getattr(exp, name, None)
        for name in (
            "CurrentUser", "SessionUser", "CurrentRole", "CurrentVersion", "CurrentDatabase",
            "CurrentCatalog", "CurrentSchema", "CurrentSchemas",
        )
    ) if cls is not None
)
# The same information spelled as a bare keyword, which parses as a column
_ENVIRONMENT_KEYWORDS = {
    "user", "current_user", "session_user", "current_role", "current_catalog", "current_schema",
}

# Column each scoped table is filtered on
_SCOPE_KEY = {t: "tenant_id" for t in TENANT_SCOPED_TABLES} | {"tenants": "id"}

def _resolves_to_cte(table_node: exp.Table) -> bool:
    """
    True if an enclosing WITH clause defines this name, i.e. the reference is
    to a CTE in scope (mirrors SQL scoping; a CTE defined elsewhere in the
    query does not hide a real table here).
    """
    if table_node.db or table_node.catalog:
        return False
    name = table_node.name.lower()
    node = table_node.parent
    while node is not None:
        with_ = node.args.get("with_") if isinstance(node, exp.Query) else None
        if with_ and any(c.alias_or_name.lower() == name for c in with_.expressions):
            return True
        node = node.parent
    return False


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
    sql: str,
    tenant_id: int,
    dialect: str = "sqlite",
    schema: dict[str, list[str]] | None = None,
) -> VerificationResult:
    """
    Parse *sql*, apply the three symbolic guards, and scope every table to
    *tenant_id*.

    *schema* (lower-case table -> columns, every table scoped on ``tenant_id``)
    replaces the application schema, e.g. for tenant-extended BIRD databases.

    Returns a ``VerificationResult`` with the safe, tenant-scoped SQL or an
    explicit error message naming the offending element.
    """
    tenant_id = int(tenant_id)
    allowed_schemas = _DEFAULT_SCHEMAS[dialect]
    tables = TABLE_COLUMNS if schema is None else schema
    scope_key = (
        _SCOPE_KEY if schema is None
        else {t: "tenant_id" for t, cols in schema.items() if "tenant_id" in cols}
    )

    # --- Parse ----------------------------------------------------------------
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
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
    env = next(parsed.find_all(*_ENVIRONMENT_NODES), None)
    if env is not None:
        return VerificationResult(
            valid=False, sql=sql, error=f"Function not allowed: '{env.sql(dialect=dialect)}'."
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
    # alias/name → real table names (an alias may be reused in different scopes)
    referenced_tables: dict[str, set[str]] = {}
    real_table_nodes: list[exp.Table] = []

    shadowing = {
        n for n in cte_names
        if n in tables or n.startswith(("sqlite_", "pragma_", "pg_"))
    }
    if shadowing:
        errors.append(f"CTE name shadows a real or system table: {sorted(shadowing)}.")

    for table_node in parsed.find_all(exp.Table):
        table_name = table_node.name.lower()
        if _resolves_to_cte(table_node):
            if table_node.alias:
                opaque_sources.add(table_node.alias.lower())
            continue
        if table_node.db.lower() not in allowed_schemas or table_node.catalog:
            errors.append(f"Schema-qualified table not allowed: '{table_node.sql(dialect=dialect)}'.")
            continue
        if table_name not in tables:
            errors.append(f"Hallucinated table: '{table_name or table_node.sql(dialect=dialect)}' does not exist in the schema.")
            continue
        if table_name not in scope_key:
            errors.append(f"Table '{table_name}' has no tenant scope; refusing to query it.")
            continue
        referenced_tables.setdefault((table_node.alias or table_name).lower(), set()).add(table_name)
        real_table_nodes.append(table_node)

    select_aliases = {a.alias.lower() for a in parsed.find_all(exp.Alias)}

    known_columns = {c for real in referenced_tables.values() for t in real for c in tables[t]}

    # Check columns
    for col_node in parsed.find_all(exp.Column):
        col_name = col_node.name.lower()
        tbl_ref = (col_node.table or "").lower()

        if not tbl_ref and col_name in _ENVIRONMENT_KEYWORDS and col_name not in known_columns:
            errors.append(f"Function not allowed: '{col_name}'.")
            continue
        if not tbl_ref and not referenced_tables and not opaque_sources and col_name not in select_aliases:
            errors.append(f"Hallucinated column: '{col_name}' is not a column of any table in the query.")
            continue

        if tbl_ref in opaque_sources:
            continue
        if isinstance(col_node.this, exp.Star):
            # ``t.*``: only the qualifier needs to resolve
            if tbl_ref not in referenced_tables:
                errors.append(f"Unknown table reference: '{tbl_ref}' in '{tbl_ref}.*'.")
            continue
        if tbl_ref and tbl_ref in referenced_tables:
            real = referenced_tables[tbl_ref]
            if not any(col_name in tables[t] for t in real):
                errors.append(
                    f"Hallucinated column: '{col_name}' does not exist in table '{'/'.join(sorted(real))}'. "
                    f"Valid columns: {sorted(c for t in real for c in tables[t])}"
                )
        elif tbl_ref and tbl_ref in tables:
            # Direct table reference without alias
            valid_cols = tables[tbl_ref]
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
                for t in real:
                    all_cols.update(tables[t])
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
        # No alias: reuse the table's own identifier so quoting survives ("order")
        alias = table_node.args.get("alias") or exp.TableAlias(this=table_node.this.copy())
        scoped = (
            exp.select("*")
            .from_(exp.Table(this=table_node.this.copy()))  # keeps quoting, e.g. "order"
            .where(
                exp.EQ(
                    this=exp.column(scope_key[real_table]),
                    expression=exp.Literal.number(tenant_id),
                )
            )
        )
        table_node.replace(exp.Subquery(this=scoped, alias=alias.copy()))

    mutated_sql = parsed.sql(dialect=dialect)
    return VerificationResult(valid=True, sql=mutated_sql)
