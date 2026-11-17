"""
Pytest suite for the Parameterized Semantic Cache.

Tests:
1. Entity stripping correctly extracts quoted strings and numbers.
2. Semantically similar queries with different params map to the same template.
3. Parameter rebinding produces correct SQL.
4. Mismatched parameter counts do NOT result in a cache hit.
5. Cross-tenant isolation: cache entries from tenant 1 don't match tenant 2.
"""

import pytest

from app.cache import ParameterizedSemanticCache, strip_entities


class TestStripEntities:
    """Test the entity-stripping normalisation."""

    def test_quoted_strings(self):
        normalised, params = strip_entities("Show students in 'CSE' department")
        assert "CSE" in params
        assert "__PARAM_0__" in normalised
        assert "CSE" not in normalised

    def test_double_quoted_strings(self):
        normalised, params = strip_entities('Find students in "ECE"')
        assert "ECE" in params
        assert "__PARAM_0__" in normalised

    def test_numeric_literals(self):
        normalised, params = strip_entities("Students with cgpa > 8.5")
        assert "8.5" in params
        assert "__PARAM_0__" in normalised

    def test_mixed_entities(self):
        normalised, params = strip_entities(
            "Show students in 'CSE' with cgpa > 8.5"
        )
        assert len(params) == 2
        assert "CSE" in params
        assert "8.5" in params
        assert "__PARAM_0__" in normalised
        assert "__PARAM_1__" in normalised

    def test_no_entities(self):
        normalised, params = strip_entities("Show all students")
        assert params == []
        assert normalised == "Show all students"


class TestParameterizedCache:
    """Test cache lookup, rebinding, and isolation."""

    @pytest.fixture
    def cache(self):
        c = ParameterizedSemanticCache(threshold=0.80)  # Lower for test stability
        return c

    def test_store_and_hit(self, cache):
        """Store a query and verify it hits on an identical query."""
        query = "Show students in 'CSE' with cgpa > 8.5"
        sql = "SELECT * FROM students WHERE department = 'CSE' AND cgpa > 8.5"
        cache.store(query, sql, tenant_id=1)
        assert cache.size == 1

        result_sql, hit = cache.lookup(query, tenant_id=1)
        assert hit is True
        assert result_sql is not None

    def test_semantic_variation_rebind(self, cache):
        """
        Two queries with the same intent but different params should
        match and rebind correctly.
        """
        # Store the first variant
        query1 = "Show students in 'CSE' with cgpa > 8.5"
        sql1 = "SELECT * FROM students WHERE department = 'CSE' AND cgpa > 8.5"
        cache.store(query1, sql1, tenant_id=1)

        # Lookup with a semantically similar but parametrically different query
        query2 = "Show students in 'ECE' with cgpa > 9.0"
        result_sql, hit = cache.lookup(query2, tenant_id=1)

        assert hit is True
        assert result_sql is not None
        # Verify params were rebound
        assert "ECE" in result_sql
        assert "9.0" in result_sql
        # Old params should NOT be present
        assert "CSE" not in result_sql

    def test_param_count_mismatch_no_hit(self, cache):
        """Mismatched parameter counts should prevent a cache hit."""
        query1 = "Show students in 'CSE' with cgpa > 8.5"
        sql1 = "SELECT * FROM students WHERE department = 'CSE' AND cgpa > 8.5"
        cache.store(query1, sql1, tenant_id=1)

        # Query with different number of params
        query2 = "Show all students"
        result_sql, hit = cache.lookup(query2, tenant_id=1)
        assert hit is False

    def test_cross_tenant_isolation(self, cache):
        """Cache entries from tenant 1 should NOT match queries from tenant 2."""
        query = "Show students in 'CSE' with cgpa > 8.5"
        sql = "SELECT * FROM students WHERE department = 'CSE' AND cgpa > 8.5"
        cache.store(query, sql, tenant_id=1)

        result_sql, hit = cache.lookup(query, tenant_id=2)
        assert hit is False

    def test_unquoted_entity_change_is_a_miss(self, cache):
        """A hard-coded literal the new question doesn't mention must not be reused."""
        cache.store(
            "Show students in CSE with cgpa > 8.5",
            "SELECT * FROM students WHERE department = 'CSE' AND cgpa > 8.5",
            tenant_id=1,
        )
        result_sql, hit = cache.lookup("Show students in ECE with cgpa > 9.0", tenant_id=1)
        assert hit is False
        assert result_sql is None

    def test_tenant_literal_never_parameterised(self, cache):
        """A param equal to the tenant id must not rebind the tenant predicate."""
        cache.store(
            "Show top 1 students by cgpa",
            "SELECT * FROM students WHERE tenant_id = 1 ORDER BY cgpa DESC LIMIT 1",
            tenant_id=1,
        )
        result_sql, hit = cache.lookup("Show top 2 students by cgpa", tenant_id=1)
        assert hit is True
        assert "tenant_id = 1" in result_sql
        assert "LIMIT 2" in result_sql

    def test_rebound_string_is_escaped(self, cache):
        """Quoted params are bound as escaped SQL literals, not spliced as text."""
        cache.store(
            "Show students in 'CSE'",
            "SELECT * FROM students WHERE department = 'CSE'",
            tenant_id=1,
        )
        result_sql, hit = cache.lookup("""Show students in "x' OR '1'='1\"""", tenant_id=1)
        assert hit is True
        assert "'x'' OR ''1''=''1'" in result_sql

    def test_clear(self, cache):
        """Cache clear should remove all entries."""
        cache.store("students in 'X' above 5", "SELECT * FROM students WHERE department = 'X' AND cgpa > 5", tenant_id=1)
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0


class TestRepeatedConstants:
    """A constant the model uses twice must be rebound twice (or not cached)."""

    def test_repeated_literal_is_rebound_everywhere(self):
        cache = ParameterizedSemanticCache()
        cache.store(
            "Show assessments with score > 80",
            "SELECT * FROM assessments WHERE score > 80 AND id IN (SELECT id FROM assessments WHERE score > 80)",
            tenant_id=1,
        )
        sql, hit = cache.lookup("Show assessments with score > 90", tenant_id=1)
        assert hit and "80" not in sql and sql.count("90") == 2

    def test_equal_parameters_are_not_cached(self):
        cache = ParameterizedSemanticCache()
        cache.store(
            "Show students with cgpa > 8 and year > 8",
            "SELECT * FROM students WHERE cgpa > 8 AND year > 8",
            tenant_id=1,
        )
        assert cache.size == 0

    def test_sql_that_ignores_a_question_constant_is_not_cached(self):
        cache = ParameterizedSemanticCache()
        cache.store("Show students in 'CSE'", "SELECT * FROM students", tenant_id=1)
        assert cache.size == 0


class TestGroundingIsWholeWord:
    """Regression: substring grounding let 'active' be reused for "inactive"."""

    def test_fixed_literal_inside_another_word_is_a_miss(self):
        cache = ParameterizedSemanticCache()
        cache.store(
            "Count total active students",
            "SELECT COUNT(*) FROM students WHERE status = 'active'",
            tenant_id=1,
        )
        _, hit = cache.lookup("Count total inactive students", tenant_id=1)
        assert not hit

    def test_fixed_literal_as_word_still_hits(self):
        cache = ParameterizedSemanticCache()
        cache.store(
            "Count active students with cgpa > 8",
            "SELECT COUNT(*) FROM students WHERE status = 'active' AND cgpa > 8",
            tenant_id=1,
        )
        sql, hit = cache.lookup("Count active students with cgpa > 9", tenant_id=1)
        assert hit and "cgpa > 9" in sql and "'active'" in sql


def test_cache_hit_reads_rows_written_after_the_template_was_stored():
    """Freshness: a hit re-executes SQL, so a row added between requests is returned."""
    from app.database import get_connection, reset_connection
    from app.pipeline import Pipeline

    reset_connection()
    try:
        pipe = Pipeline(use_mock_llm=True)
        first = pipe.run("Show students with cgpa > 8", tenant_id=1, mode="ast_guardcache")
        assert first.error is None and not first.cache_hit
        before = pipe.run("Show students with cgpa > 9", tenant_id=1, mode="ast_guardcache")
        assert before.cache_hit and "Fresh Student" not in {r["name"] for r in before.execution_rows}

        conn = get_connection()  # privileged write, outside the pipeline
        conn.execute("PRAGMA query_only = OFF")
        conn.execute(
            "INSERT INTO students (tenant_id, name, email, department, cgpa, year) "
            "VALUES (1, 'Fresh Student', 'fresh@stride.edu', 'CSE', 9.8, 2)"
        )
        conn.execute("PRAGMA query_only = ON")

        after = pipe.run("Show students with cgpa > 9.5", tenant_id=1, mode="ast_guardcache")
        assert after.cache_hit and "Fresh Student" in {r["name"] for r in after.execution_rows}
    finally:
        reset_connection()


def test_wide_schema_is_opt_in():
    import subprocess
    import sys

    from app.database import TABLE_DDLS, WIDE_DDLS, WIDE_FK_GRAPH

    assert len(TABLE_DDLS) == 10 and len(WIDE_DDLS) == 50
    assert all(fk in WIDE_DDLS or fk == "tenants" for fks in WIDE_FK_GRAPH.values() for fk in fks)
    code = ("from app.database import TABLE_DDLS, get_connection;"
            "c = get_connection(); print(len(TABLE_DDLS), c.execute('SELECT COUNT(*) FROM payslips').fetchone()[0])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "WIDE_SCHEMA": "1"})
    assert out.stdout.split() == ["60", "0"]


class TestEqualValuesInDifferentRoles:
    """One question constant must not be bound to SQL literals that play different roles."""

    def test_filter_value_and_limit_are_not_merged(self):
        cache = ParameterizedSemanticCache()
        cache.store(
            "Show the top 1 students in year one by cgpa",
            "SELECT * FROM students WHERE year = 1 ORDER BY cgpa DESC LIMIT 1",
            tenant_id=1,
        )
        assert cache.size == 0  # otherwise "top 3" would be rebound to year = 3 as well

    def test_same_role_twice_is_still_cached(self):
        cache = ParameterizedSemanticCache()
        cache.store(
            "List students with cgpa above 8.0 and the number of students above that cgpa",
            "SELECT name, (SELECT COUNT(*) FROM students WHERE cgpa > 8.0) AS n FROM students WHERE cgpa > 8.0",
            tenant_id=1,
        )
        assert cache.size == 1
