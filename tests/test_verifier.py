"""
Pytest suite for the Symbolic AST Verifier.

Tests:
1. DROP TABLE is blocked (read-only guard).
2. Hallucinated columns trigger specific error feedback.
3. tenant_id is deterministically injected into single-table queries.
4. tenant_id is injected into JOIN queries for all tenant-scoped tables.
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
        # Both aliases should have tenant_id injected
        assert "s.tenant_id = 1" in sql_lower
        assert "a.tenant_id = 1" in sql_lower

    def test_tenants_table_no_injection(self):
        """The `tenants` table itself should NOT get tenant_id injected."""
        result = verify_and_inject_ast(
            "SELECT * FROM tenants", tenant_id=1
        )
        assert result.valid
        # The tenants table doesn't have tenant_id column
        assert "tenant_id" not in result.sql.lower()


class TestSQLParseError:
    """Ensure malformed SQL is handled gracefully."""

    def test_garbage_sql(self):
        result = verify_and_inject_ast("THIS IS NOT SQL", tenant_id=1)
        # sqlglot may parse this or error — either way it shouldn't crash
        assert isinstance(result.valid, bool)
