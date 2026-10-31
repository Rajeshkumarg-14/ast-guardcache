"""
Hybrid Schema-Graph Retriever.

Combines BM25 lexical matching with dense sentence-transformer embeddings via
Reciprocal Rank Fusion (RRF), then expands the result set with 1-hop FK closure
to guarantee JOIN-able context for the LLM.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.database import FK_GRAPH, TABLE_DDLS

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def _tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics (so notification_queue → notification, queue)
    and drop a trailing 's' as a minimal plural fold."""
    # ponytail: naive plural strip, swap for a real stemmer if vocab grows
    return [w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower())]


class HybridSchemaRetriever:
    """
    Retrieves the most relevant table DDLs for a natural-language question.

    Pipeline:
    1. BM25 ranking over tokenized DDL strings.
    2. Dense cosine-similarity ranking using all-MiniLM-L6-v2.
    3. Reciprocal Rank Fusion (RRF) to merge both ranked lists.
    4. 1-hop FK expansion to include related tables.
    """

    def __init__(self) -> None:
        self.table_names: list[str] = list(TABLE_DDLS.keys())
        self.ddl_strings: list[str] = [TABLE_DDLS[t] for t in self.table_names]

        # BM25 index
        tokenized = [_tokenize(ddl) for ddl in self.ddl_strings]
        self.bm25 = BM25Okapi(tokenized)

        # Dense index
        model = _get_model()
        self.embeddings: np.ndarray = model.encode(
            self.ddl_strings, normalize_embeddings=True
        )

    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[str], list[str]]:
        """
        Return ``(table_names, ddl_strings)`` relevant to *question*.

        Includes 1-hop FK neighbours beyond the initial top-K.
        """
        if top_k is None:
            top_k = settings.RETRIEVER_TOP_K

        # --- BM25 scores → ranks ------------------------------------------
        bm25_scores = self.bm25.get_scores(_tokenize(question))
        bm25_order = np.argsort(-bm25_scores)  # descending

        # --- Dense scores → ranks -----------------------------------------
        model = _get_model()
        q_emb = model.encode([question], normalize_embeddings=True)
        dense_scores = (self.embeddings @ q_emb.T).squeeze()
        dense_order = np.argsort(-dense_scores)

        # --- Reciprocal Rank Fusion ----------------------------------------
        k = settings.RRF_K
        rrf_scores: dict[int, float] = {}
        # 1-based ranks, as in Cormack et al. and paper Eq. (2)
        for rank, idx in enumerate(bm25_order, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, idx in enumerate(dense_order, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
        top_indices = sorted_indices[:top_k]

        # --- 1-hop FK closure ---------------------------------------------
        selected: set[str] = set()
        for idx in top_indices:
            tbl = self.table_names[idx]
            selected.add(tbl)
            for neighbour in FK_GRAPH.get(tbl, []):
                selected.add(neighbour)

        # Preserve RRF order, appending FK neighbours at the end
        ordered_names: list[str] = []
        for idx in top_indices:
            tbl = self.table_names[idx]
            if tbl not in ordered_names:
                ordered_names.append(tbl)
        for tbl in selected:
            if tbl not in ordered_names:
                ordered_names.append(tbl)

        ordered_ddls = [TABLE_DDLS[t] for t in ordered_names]
        return ordered_names, ordered_ddls
