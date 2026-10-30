"""
Parameterized SQL-Template Semantic Cache.

Core insight: two queries like
  "Show students in CSE with cgpa > 8.5"
  "List students in ECE with cgpa > 9.0"
share the same *intent* once entity literals (CSE, 8.5, ECE, 9.0) are stripped.

Pipeline:
1. strip_entities(query)  → normalised template string + ordered params list.
2. Embed the normalised template.
3. On lookup, find the nearest cached template by cosine similarity (≥ 0.88).
4. If parameter counts match, bind the new params into the cached SQL template.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Entity stripping
# ---------------------------------------------------------------------------

# Patterns: quoted strings (single or double), then standalone numbers (int/float)
_QUOTED_RE = re.compile(r"""(['"])(.*?)\1""")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def strip_entities(query: str) -> tuple[str, list[str]]:
    """
    Extract literal entities from a natural-language query.

    Returns ``(normalised_template, params)`` where each literal is replaced
    with a positional placeholder ``__PARAM_N__``.
    """
    params: list[str] = []
    counter = 0

    def _replace_quoted(m: re.Match) -> str:
        nonlocal counter
        params.append(m.group(2))
        placeholder = f"__PARAM_{counter}__"
        counter += 1
        return placeholder

    normalised = _QUOTED_RE.sub(_replace_quoted, query)

    def _replace_number(m: re.Match) -> str:
        nonlocal counter
        params.append(m.group(0))
        placeholder = f"__PARAM_{counter}__"
        counter += 1
        return placeholder

    normalised = _NUMBER_RE.sub(_replace_number, normalised)
    return normalised, params


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Single cached SQL template with its parameter bindings."""

    normalised_query: str
    embedding: np.ndarray
    sql_template: str  # SQL with __PARAM_N__ placeholders
    params: list[str]
    tenant_id: int


# ---------------------------------------------------------------------------
# Parameterised Semantic Cache
# ---------------------------------------------------------------------------


class ParameterizedSemanticCache:
    """
    In-memory vector cache that matches parameterised intent templates
    and rebinds fresh entity values into previously-verified SQL.
    """

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold or settings.CACHE_SIMILARITY_THRESHOLD
        self._entries: list[_CacheEntry] = []

    # ----- public API ------------------------------------------------------

    def lookup(
        self, query: str, tenant_id: int
    ) -> tuple[str | None, bool]:
        """
        Check the cache for a matching intent template.

        Returns ``(rebound_sql, cache_hit)``.  If no match, returns
        ``(None, False)``.
        """
        if not self._entries:
            return None, False

        normalised, new_params = strip_entities(query)
        model = _get_model()
        q_emb = model.encode([normalised], normalize_embeddings=True)

        # Build matrix of cached embeddings
        cached_embs = np.stack([e.embedding for e in self._entries])
        sims = (cached_embs @ q_emb.T).squeeze()

        if sims.ndim == 0:
            sims = np.array([sims])

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self.threshold:
            return None, False

        entry = self._entries[best_idx]

        # Parameter count must match for safe rebinding
        if len(new_params) != len(entry.params):
            return None, False

        # Tenant must match (cache is tenant-scoped)
        if entry.tenant_id != tenant_id:
            return None, False

        # Rebind parameters
        rebound = entry.sql_template
        for i, param in enumerate(new_params):
            rebound = rebound.replace(f"__PARAM_{i}__", param)

        return rebound, True

    def store(
        self, query: str, sql: str, tenant_id: int
    ) -> None:
        """
        Store a verified SQL alongside its parameterised intent template.

        The SQL is converted to a template by replacing the *original*
        literal values with positional placeholders.
        """
        normalised, params = strip_entities(query)
        model = _get_model()
        emb = model.encode([normalised], normalize_embeddings=True)[0]

        # Build the SQL template by replacing original param values
        sql_template = sql
        for i, param in enumerate(params):
            # Replace first occurrence to avoid over-replacing
            sql_template = sql_template.replace(param, f"__PARAM_{i}__", 1)

        self._entries.append(
            _CacheEntry(
                normalised_query=normalised,
                embedding=emb,
                sql_template=sql_template,
                params=params,
                tenant_id=tenant_id,
            )
        )

    def clear(self) -> None:
        """Flush all cached entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
