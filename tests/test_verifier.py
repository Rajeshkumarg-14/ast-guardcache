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


class TestBirdFalseRejections:
    """Valid SQL from the BIRD dev set that earlier versions rejected or broke."""

    SCHEMA = {
        "order": ["tenant_id", "order_id", "account_id"],
        "account": ["tenant_id", "account_id", "frequency"],
    }

    def test_reserved_word_table_stays_quoted(self):
        import sqlite3

        result = verify_and_inject_ast(
            "SELECT T1.order_id FROM `order` AS T1 JOIN account AS T2 ON T1.account_id = T2.account_id",
            tenant_id=1,
            schema=self.SCHEMA,
        )
        assert result.valid, result.error
        conn = sqlite3.connect(":memory:")
        conn.execute('CREATE TABLE "order" (tenant_id, order_id, account_id)')
        conn.execute("CREATE TABLE account (tenant_id, account_id, frequency)")
        conn.execute(result.sql)  # raised "near \"order\": syntax error" before

    def test_unaliased_reserved_word_table_keeps_quoted_alias(self):
        import sqlite3

        result = verify_and_inject_ast(
            "SELECT account_id FROM `order` GROUP BY account_id", tenant_id=1, schema=self.SCHEMA
        )
        assert result.valid, result.error
        conn = sqlite3.connect(":memory:")
        conn.execute('CREATE TABLE "order" (tenant_id, order_id, account_id)')
        conn.execute(result.sql)  # raised: generated alias was a bare ``order``

    def test_datetime_allowed(self):
        assert verify_and_inject_ast("SELECT DATETIME() AS now, name FROM students", tenant_id=1).valid

    def test_cte_referenced_under_alias(self):
        sql = "WITH c AS (SELECT department FROM students) SELECT mb.department FROM c AS mb"
        assert verify_and_inject_ast(sql, tenant_id=1).valid

    def test_alias_reused_across_scopes(self):
        sql = (
            "SELECT T1.name FROM students AS T1 JOIN (SELECT MAX(cgpa) AS top FROM students) AS T2 "
            "ON T1.cgpa = T2.top WHERE T1.id IN (SELECT T2.student_id FROM assessments AS T2)"
        )
        result = verify_and_inject_ast(sql, tenant_id=1)
        assert result.valid, result.error
        assert result.sql.lower().count("where tenant_id = 1") == 3

    def test_fake_column_still_rejected_with_reused_alias(self):
        sql = "SELECT T1.salary FROM students AS T1 WHERE T1.id IN (SELECT T1.id FROM assessments AS T1)"
        assert not verify_and_inject_ast(sql, tenant_id=1).valid


class TestQualifiedStar:
    """``table.*`` / ``alias.*`` is valid SQL and must not be read as a column named '*'."""

    def test_table_star(self):
        result = verify_and_inject_ast("SELECT students.* FROM students WHERE cgpa > 8", tenant_id=1)
        assert result.valid, result.error

    def test_alias_star_in_join_is_scoped(self):
        result = verify_and_inject_ast(
            "SELECT s.* FROM students s JOIN tenants t ON s.tenant_id = t.id", tenant_id=1
        )
        assert result.valid, result.error
        assert result.sql.count("tenant_id = 1") == 1 and "id = 1" in result.sql

    def test_star_on_unknown_qualifier_rejected(self):
        result = verify_and_inject_ast("SELECT x.* FROM students", tenant_id=1)
        assert not result.valid and "x" in result.error


class TestEnvironmentDisclosure:
    """Queries that reveal the server (user, version, database) are not tenant data."""

    @pytest.mark.parametrize("sql", [
        "SELECT current_user", "SELECT session_user", "SELECT user", "SELECT current_role",
        "SELECT version()", "SELECT current_database()", "SELECT current_schema()",
        "SELECT current_catalog", "SELECT name, current_user FROM students",
        "SELECT current_role FROM (SELECT 1 AS x) AS d",
    ])
    def test_postgres_environment_rejected(self, sql):
        assert not verify_and_inject_ast(sql, tenant_id=1, dialect="postgres").valid

    def test_sqlite_version_rejected(self):
        assert not verify_and_inject_ast("SELECT sqlite_version()", tenant_id=1).valid

    def test_bare_column_without_table_rejected(self):
        assert not verify_and_inject_ast("SELECT secret", tenant_id=1).valid

    def test_constants_and_dates_still_allowed(self):
        assert verify_and_inject_ast("SELECT 1 AS one", tenant_id=1).valid
        assert verify_and_inject_ast("SELECT name FROM students WHERE created_at < CURRENT_DATE", tenant_id=1).valid


class TestResultComparison:
    """The isolation oracle must not call float noise or row order a different result."""

    def test_float_noise_and_order_are_ignored(self):
        from app.database import canon_rows

        a = [{"track": "ML", "avg": 76.63500000000002}, {"track": "DB", "avg": 66.92866666666667}]
        b = [{"track": "DB", "avg": 66.92866666666666}, {"track": "ML", "avg": 76.635}]
        assert canon_rows(a) == canon_rows(b)

    def test_real_differences_are_kept(self):
        from app.database import canon_rows

        assert canon_rows([{"avg": 76.635}]) != canon_rows([{"avg": 76.636}])
        assert canon_rows([(1, "a")]) != canon_rows([(1, "a"), (1, "a")])


class TestScoring:
    """Benchmark scoring rules (scripts/run_benchmark.py)."""

    def test_empty_output_is_not_an_answer(self):
        from scripts.run_benchmark import executed

        assert not executed("", None) and not executed("   ", None) and not executed(None, None)
        assert not executed("SELECT 1", "Execution error: x")
        assert executed("SELECT 1", None)

    def test_duplicate_column_names_keep_both_values(self):
        from app.database import execute_sql

        row = execute_sql(
            "SELECT s.name, ct.name FROM students s JOIN assessments a ON a.student_id = s.id "
            "JOIN competency_tracks ct ON ct.id = a.track_id LIMIT 1"
        )[0]
        assert list(row) == ["name", "name_2"] and row["name"] != row["name_2"]

    def test_row_matching_is_a_true_matching(self):
        from scripts.run_benchmark import check_execution_accuracy

        # gold rows (1, 2) and (1, 3); predicted (1, 2, 3) and (1, 2, 4).
        # First-fit gives (1, 2) the row (1, 2, 3) and then fails; a matching exists.
        gold = "SELECT 1 AS a, 2 AS b UNION ALL SELECT 1, 3"
        pred = [{"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 4}]
        assert check_execution_accuracy(pred, gold)
        assert not check_execution_accuracy([{"a": 1, "b": 2, "c": 4}, {"a": 1, "b": 2, "c": 4}], gold)
