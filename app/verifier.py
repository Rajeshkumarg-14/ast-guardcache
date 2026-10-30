"""
Symbolic AST Verifier & Tenant RLS Injector.

Uses sqlglot to parse candidate SQL and enforce three symbolic invariants:
1. Read-Only Guard       – only SELECT statements allowed.
2. Schema Existence Guard – every table/column must exist in the metadata.
3. Tenant RLS Injection   – WHERE tenant_id = ? appended to every tenant-scoped table.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from app.database import TABLE_COLUMNS, TENANT_SCOPED_TABLES

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
    Parse *sql*, apply the three symbolic guards, and inject tenant_id RLS.

    Returns a ``VerificationResult`` with the safe, tenant-scoped SQL or an
    explicit error message naming the offending element.
    """
    # --- Parse ----------------------------------------------------------------
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        return VerificationResult(valid=False, sql=sql, error=f"SQL parse error: {exc}")

    # --- 1. Read-Only Guard ---------------------------------------------------
    if not isinstance(parsed, exp.Select):
        stmt_type = type(parsed).__name__
        return VerificationResult(
            valid=False,
            sql=sql,
            error=f"Read-only violation: {stmt_type} statements are not allowed. Only SELECT queries are permitted.",
        )

    # --- 2. Schema Existence Guard -------------------------------------------
    errors: list[str] = []
    referenced_tables: dict[str, str] = {}  # alias/name → real table name

    for table_node in parsed.find_all(exp.Table):
        table_name = table_node.name.lower()
        alias = (table_node.alias or table_name).lower()

        if table_name not in TABLE_COLUMNS:
            errors.append(f"Hallucinated table: '{table_name}' does not exist in the schema.")
            continue

        referenced_tables[alias] = table_name

    # Check columns
    for col_node in parsed.find_all(exp.Column):
        col_name = col_node.name.lower()
        tbl_ref = (col_node.table or "").lower()

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
            referenced_tables[tbl_ref] = tbl_ref
        elif not tbl_ref:
            # Unqualified column — check if it exists in any referenced table
            found = any(
                col_name in TABLE_COLUMNS.get(real, [])
                for real in referenced_tables.values()
            )
            if not found and referenced_tables:
                # Only flag if we have context to validate against
                all_cols = set()
                for real in referenced_tables.values():
                    all_cols.update(TABLE_COLUMNS.get(real, []))
                if col_name not in all_cols:
                    errors.append(
                        f"Hallucinated column: '{col_name}' not found in any referenced table."
                    )

    if errors:
        return VerificationResult(
            valid=False,
            sql=sql,
            error=" | ".join(errors),
        )

    # --- 3. Tenant RLS Injection ----------------------------------------------
    for alias, real_table in referenced_tables.items():
        if real_table in TENANT_SCOPED_TABLES:
            tenant_condition = f"{alias}.tenant_id = {tenant_id}"
            parsed = parsed.where(tenant_condition, append=True)

    mutated_sql = parsed.sql(dialect="sqlite")
    return VerificationResult(valid=True, sql=mutated_sql)
