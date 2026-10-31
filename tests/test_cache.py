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
        cache.store("test query 'X' with 5", "SELECT 1", tenant_id=1)
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0
