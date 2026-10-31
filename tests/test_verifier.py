"""
Pytest suite for the Symbolic AST Verifier.

Tests:
1. DROP TABLE is blocked (read-only guard).
2. Hallucinated columns trigger specific error feedback.
3. Every table reference is replaced by a tenant-scoped derived table.
4. Scoping applies in every scope (joins, subqueries) and to `tenants` via id.
5. INSERT/UPDATE/DELETE are blocked.
"""

import pytest

from app.verifier import verify_and_inject_ast


class TestReadOnlyGuard:
    """Ensure non-SELECT statements are rejected."""

    def test_drop_table_blocked(self):
        result = verify_and_inject_ast("DROP TABLE students", tenant_id=1)
        assert not result.valid
        assert "Read-only violation" in result.error
        assert "Drop" in result.error

    def test_delete_blocked(self):
        result = verify_and_inject_ast(
            "DELETE FROM students WHERE id = 1", tenant_id=1
        )
        assert not result.valid
        assert "Read-only violation" in result.error

    def test_update_blocked(self):
        result = verify_and_inject_ast(
            "UPDATE students SET cgpa = 10 WHERE id = 1", tenant_id=1
        )
        assert not result.valid
        assert "Read-only violation" in result.error

    def test_insert_blocked(self):
        result = verify_and_inject_ast(
            "INSERT INTO students (tenant_id, name, email, department, cgpa, year) "
            "VALUES (1, 'Test', 'test@test.com', 'CSE', 9.0, 1)",
            tenant_id=1,
        )
        assert not result.valid
        assert "Read-only violation" in result.error


class TestSchemaExistenceGuard:
    """Ensure hallucinated tables/columns are caught."""

    def test_hallucinated_table(self):
        result = verify_and_inject_ast(
            "SELECT * FROM nonexistent_table", tenant_id=1
        )
        assert not result.valid
        assert "Hallucinated table" in result.error
        assert "nonexistent_table" in result.error

    def test_hallucinated_column(self):
        result = verify_and_inject_ast(
            "SELECT students.nonexistent_col FROM students", tenant_id=1
        )
        assert not result.valid
        assert "Hallucinated column" in result.error
        assert "nonexistent_col" in result.error

    def test_valid_table_and_columns(self):
        result = verify_and_inject_ast(
            "SELECT students.name, students.cgpa FROM students", tenant_id=1
        )
        assert result.valid


class TestTenantInjection:
    """Ensure tenant_id WHERE clauses are injected."""

    def test_single_table_injection(self):
        result = verify_and_inject_ast(
            "SELECT * FROM students", tenant_id=1
        )
        assert result.valid
        sql_lower = result.sql.lower()
        assert "tenant_id" in sql_lower
        assert "= 1" in sql_lower

    def test_single_table_different_tenant(self):
        result = verify_and_inject_ast(
            "SELECT * FROM students", tenant_id=2
        )
        assert result.valid
        sql_lower = result.sql.lower()
        assert "tenant_id" in sql_lower
        assert "= 2" in sql_lower

    def test_join_query_injection(self):
        """Both tables in a JOIN should get tenant_id filters."""
        sql = (
            "SELECT s.name, a.score "
            "FROM students AS s "
            "JOIN assessments AS a ON s.id = a.student_id"
        )
        result = verify_and_inject_ast(sql, tenant_id=1)
        assert result.valid
        sql_lower = result.sql.lower()
        # Both tables are replaced by tenant-scoped derived tables
        assert "(select * from students where tenant_id = 1) as s" in sql_lower
        assert "(select * from assessments where tenant_id = 1) as a" in sql_lower

    def test_tenants_table_scoped_on_id(self):
        """The `tenants` table is scoped to the caller's own row via id."""
        result = verify_and_inject_ast(
            "SELECT * FROM tenants", tenant_id=1
        )
        assert result.valid
        assert "where id = 1" in result.sql.lower()

    def test_subquery_is_scoped(self):
        """Scalar subqueries get their own tenant scope (no cross-tenant AVG)."""
        result = verify_and_inject_ast(
            "SELECT name, (SELECT AVG(cgpa) FROM students) AS g FROM students",
            tenant_id=1,
        )
        assert result.valid
        assert result.sql.lower().count("where tenant_id = 1") == 2


class TestSQLParseError:
    """Ensure malformed SQL is handled gracefully."""

    def test_garbage_sql(self):
        result = verify_and_inject_ast("THIS IS NOT SQL", tenant_id=1)
        # sqlglot may parse this or error — either way it shouldn't crash
        assert isinstance(result.valid, bool)
