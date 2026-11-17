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

Templates are built from the *pre-injection* SQL by rewriting sqlglot literal
nodes (never by substring replacement), so tenant predicates are never
parameterised. Callers must re-verify rebound SQL before executing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import sqlglot
from sqlglot import exp

from app.config import settings

# Shared with the retriever so the embedding model is loaded once per process
from app.retriever import _get_model  # noqa: E402


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
# Operator signature
# ---------------------------------------------------------------------------

# Sentence embeddings barely separate "score > 80" from "score < 80" or "top 5"
# from "bottom 5", but the SQL differs. The ordered sequence of comparator,
# sort-direction and negation cues must match exactly for a cache hit.
_SIGNATURE_PHRASES = [
    ("no more than", "le"), ("no less than", "ge"), ("at least", "ge"), ("at most", "le"),
    ("or more", "ge"), ("or less", "le"), ("or above", "ge"), ("or below", "le"),
    ("greater than or equal to", "ge"), ("less than or equal to", "le"),
    ("more than", "gt"), ("greater than", "gt"), ("less than", "lt"), ("fewer than", "lt"),
    ("not equal to", "ne"), ("equal to", "eq"),
]
_SIGNATURE_WORDS = {
    ">=": "ge", "<=": "le", "!=": "ne", "<>": "ne", ">": "gt", "<": "lt", "=": "eq",
    "above": "gt", "over": "gt", "exceeding": "gt", "exceeds": "gt", "greater": "gt",
    "more": "gt", "higher": "gt", "after": "gt",
    "below": "lt", "under": "lt", "less": "lt", "fewer": "lt", "lower": "lt", "before": "lt",
    "equals": "eq", "exactly": "eq",
    "top": "hi", "highest": "hi", "best": "hi", "most": "hi", "max": "hi", "maximum": "hi",
    "largest": "hi", "latest": "hi", "newest": "hi", "descending": "hi", "desc": "hi",
    "bottom": "lo", "lowest": "lo", "worst": "lo", "least": "lo", "min": "lo", "minimum": "lo",
    "smallest": "lo", "earliest": "lo", "oldest": "lo", "ascending": "lo", "asc": "lo",
    "not": "neg", "no": "neg", "without": "neg", "except": "neg", "excluding": "neg",
    "never": "neg", "non": "neg",
}
_SIGNATURE_TOKEN_RE = re.compile(r">=|<=|!=|<>|[<>=]|[a-z]+")


def operator_signature(query: str) -> tuple[str, ...]:
    """Ordered comparator / direction / negation cues in *query*."""
    text = query.lower()
    for phrase, tag in _SIGNATURE_PHRASES:
        text = re.sub(rf"\b{phrase}\b", f" zzsig{tag} ", text)
    out: list[str] = []
    for tok in _SIGNATURE_TOKEN_RE.findall(text):
        if tok.startswith("zzsig"):
            out.append(tok[5:])
        elif tok in _SIGNATURE_WORDS:
            out.append(_SIGNATURE_WORDS[tok])
    return tuple(out)


# ---------------------------------------------------------------------------
# SQL template helpers
# ---------------------------------------------------------------------------


def _is_tenant_literal(node: exp.Literal) -> bool:
    """True if the literal is the value side of a ``tenant_id`` comparison."""
    parent = node.parent
    return isinstance(parent, exp.Binary) and any(
        isinstance(side, exp.Column) and side.name.lower() == "tenant_id"
        for side in (parent.left, parent.right)
    )


def _literal_equals(node: exp.Literal, param: str) -> bool:
    if node.is_string:
        return node.this == param
    try:
        return float(node.this) == float(param)
    except ValueError:
        return False


def _to_template(sql: str, params: list[str]) -> str | None:
    """
    Replace literals matching *params* with typed placeholders (:s0 / :n1).

    Every occurrence of a parameter value is replaced, so a constant the model
    used twice is rebound twice. Returns None (do not cache) when two
    parameters share a value, because a literal could then belong to either;
    when a parameter does not occur in the SQL at all; or when one parameter
    matches literals in different roles (``year = 1 ... LIMIT 1``), because
    equal values need not be the same constant.
    """
    if any(
        _same_value(params[i], params[j])
        for i in range(len(params)) for j in range(i + 1, len(params))
    ):
        return None
    tree = sqlglot.parse_one(sql, read="sqlite")
    roles: dict[int, set] = {}
    for node in list(tree.find_all(exp.Literal)):
        if _is_tenant_literal(node):
            continue
        for i, param in enumerate(params):
            if _literal_equals(node, param):
                roles.setdefault(i, set()).add(_literal_role(node))
                kind = "s" if node.is_string else "n"
                node.replace(exp.Placeholder(this=f"{kind}{i}"))
                break
    if len(roles) != len(params):
        return None  # the SQL ignores a constant of the question: rebinding would too
    if any(len(r) > 1 for r in roles.values()):
        return None  # one value in two roles: not provably the same constant
    return tree.sql(dialect="sqlite")


def _literal_role(node: exp.Literal) -> tuple[str, str | None]:
    """Where a literal sits: the kind of its parent and the column it is compared with."""
    parent = node.parent
    column = None
    if isinstance(parent, exp.Binary):
        column = next(
            (side.name.lower() for side in (parent.left, parent.right) if isinstance(side, exp.Column)), None
        )
    return type(parent).__name__, column


def _same_value(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def _bind_template(template: str, params: list[str], query: str) -> str | None:
    """
    Bind *params* into *template* as escaped literals.

    Returns None (treat as a miss) if a numeric slot gets a non-number, or if a
    hard-coded string literal in the template is not mentioned in *query* as a
    whole word (e.g. a cached ``department = 'CSE'`` reused for a question
    about ECE, or ``status = 'active'`` for a question about inactive rows).
    """
    if any(ch < " " for p in params for ch in p):
        return None  # control characters (e.g. NUL) never come from real questions
    tree = sqlglot.parse_one(template, read="sqlite")
    for node in list(tree.find_all(exp.Placeholder)):
        kind, idx = node.name[0], int(node.name[1:])
        value = params[idx]
        if kind == "n":
            if not _NUMBER_RE.fullmatch(value):
                return None
            node.replace(exp.Literal.number(value))
        else:
            node.replace(exp.Literal.string(value))

    q = query.casefold()
    for node in tree.find_all(exp.Literal):
        # Whole-word match: 'active' must not be grounded by "inactive"
        word = re.escape(node.this.strip("%").casefold())
        if node.is_string and not re.search(rf"(?<!\w){word}(?!\w)", q):
            return None
    return tree.sql(dialect="sqlite")


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Single cached SQL template with its parameter bindings."""

    normalised_query: str
    embedding: np.ndarray
    sql_template: str  # SQL with :sN / :nN typed placeholders
    params: list[str]
    tenant_id: int
    signature: tuple[str, ...] = ()


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
        # Cache is tenant-scoped: only consider this tenant's entries
        entries = [e for e in self._entries if e.tenant_id == tenant_id]
        if not entries:
            return None, False

        normalised, new_params = strip_entities(query)
        model = _get_model()
        q_emb = model.encode([normalised], normalize_embeddings=True)

        # Build matrix of cached embeddings
        cached_embs = np.stack([e.embedding for e in entries])
        sims = (cached_embs @ q_emb.T).squeeze()

        if sims.ndim == 0:
            sims = np.array([sims])

        signature = operator_signature(normalised)

        # Try every candidate above the threshold, most similar first
        for idx in np.argsort(-sims):
            if float(sims[idx]) < self.threshold:
                break
            entry = entries[int(idx)]
            # Parameter count and operator cues must match for safe rebinding
            if len(new_params) != len(entry.params) or entry.signature != signature:
                continue
            rebound = _bind_template(entry.sql_template, new_params, query)
            if rebound is not None:
                return rebound, True
        return None, False

    def store(
        self, query: str, sql: str, tenant_id: int
    ) -> None:
        """
        Store a verified SQL alongside its parameterised intent template.

        *sql* should be the pre-injection SQL (before tenant predicates are
        appended). Literal nodes matching the query's params become typed
        placeholders.
        """
        normalised, params = strip_entities(query)
        try:
            sql_template = _to_template(sql, params)
        except sqlglot.errors.ParseError:
            return
        if sql_template is None:
            return
        # Dedupe: a miss on a stored shape (e.g. grounding rejected it) must
        # not grow the cache without bound
        if any(
            e.tenant_id == tenant_id
            and e.normalised_query == normalised
            and e.sql_template == sql_template
            for e in self._entries
        ):
            return
        model = _get_model()
        emb = model.encode([normalised], normalize_embeddings=True)[0]

        self._entries.append(
            _CacheEntry(
                normalised_query=normalised,
                embedding=emb,
                sql_template=sql_template,
                params=params,
                tenant_id=tenant_id,
                signature=operator_signature(normalised),
            )
        )

    def clear(self) -> None:
        """Flush all cached entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
