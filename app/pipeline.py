"""
Pipeline Orchestrator — supports 4 ablation modes.

Modes:
- ``zero_shot``       : Full schema → LLM → raw execute (no RAG, no AST, no cache).
- ``hybrid_rag``      : Pruned schema via retriever → LLM → raw execute.
- ``rag_ast``         : Pruned schema → LLM → AST verify + tenant inject → execute.
- ``ast_guardcache``  : Cache check → pruned schema → LLM → AST verify → cache store → execute.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.cache import ParameterizedSemanticCache
from app.database import execute_sql, get_full_schema_text
from app.llm import LLMResult, generate_sql, mock_generate_sql
from app.retriever import HybridSchemaRetriever
from app.verifier import verify_and_inject_ast

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Structured telemetry from a single pipeline run."""

    mode: str
    question: str
    generated_sql: str = ""
    verified_sql: str = ""
    execution_rows: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    ast_retries: int = 0
    pruned_tables_count: int = 0
    total_tables_count: int = 10
    latency_ms: float = 0.0
    error: str | None = None
    pruned_ddl: str = ""
    full_ddl: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """
    Orchestrates the full AST-GuardCache pipeline with configurable ablation.
    """

    def __init__(self, use_mock_llm: bool = False) -> None:
        self.retriever = HybridSchemaRetriever()
        self.cache = ParameterizedSemanticCache()
        self.use_mock_llm = use_mock_llm

    def run(
        self,
        question: str,
        tenant_id: int = 1,
        mode: str = "ast_guardcache",
    ) -> PipelineResult:
        """Execute the pipeline in the specified ablation mode."""
        t0 = time.perf_counter()
        result = PipelineResult(
            mode=mode,
            question=question,
            full_ddl=get_full_schema_text(),
        )

        try:
            if mode == "zero_shot":
                self._run_zero_shot(result, question, tenant_id)
            elif mode == "hybrid_rag":
                self._run_hybrid_rag(result, question, tenant_id)
            elif mode == "rag_ast":
                self._run_rag_ast(result, question, tenant_id)
            elif mode == "ast_guardcache":
                self._run_ast_guardcache(result, question, tenant_id)
            else:
                result.error = f"Unknown mode: {mode}"
        except Exception as exc:
            result.error = str(exc)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ----- Mode implementations --------------------------------------------

    def _run_zero_shot(
        self, result: PipelineResult, question: str, tenant_id: int
    ) -> None:
        """Full schema, no RAG, no AST verification."""
        schema_ddl = get_full_schema_text()
        result.pruned_tables_count = 10  # all tables
        result.pruned_ddl = schema_ddl

        llm_result = self._call_llm(question, schema_ddl, tenant_id, verify=False)
        result.generated_sql = llm_result.sql
        result.verified_sql = llm_result.sql  # no verification
        result.ast_retries = llm_result.retries_used

        if llm_result.error:
            result.error = llm_result.error
            return

        try:
            result.execution_rows = execute_sql(llm_result.sql)
        except Exception as exc:
            result.error = f"Execution error: {exc}"

    def _run_hybrid_rag(
        self, result: PipelineResult, question: str, tenant_id: int
    ) -> None:
        """Pruned schema via RAG, no AST verification."""
        table_names, ddl_strings = self.retriever.retrieve(question)
        schema_ddl = "\n".join(ddl_strings)
        result.pruned_tables_count = len(table_names)
        result.pruned_ddl = schema_ddl

        llm_result = self._call_llm(question, schema_ddl, tenant_id, verify=False)
        result.generated_sql = llm_result.sql
        result.verified_sql = llm_result.sql
        result.ast_retries = llm_result.retries_used

        if llm_result.error:
            result.error = llm_result.error
            return

        try:
            result.execution_rows = execute_sql(llm_result.sql)
        except Exception as exc:
            result.error = f"Execution error: {exc}"

    def _run_rag_ast(
        self, result: PipelineResult, question: str, tenant_id: int
    ) -> None:
        """Pruned schema + AST verification with tenant injection."""
        table_names, ddl_strings = self.retriever.retrieve(question)
        schema_ddl = "\n".join(ddl_strings)
        result.pruned_tables_count = len(table_names)
        result.pruned_ddl = schema_ddl

        llm_result = self._call_llm(question, schema_ddl, tenant_id, verify=True)
        result.generated_sql = llm_result.sql
        result.ast_retries = llm_result.retries_used

        if llm_result.error:
            result.error = llm_result.error
            result.verified_sql = llm_result.sql
            return

        result.verified_sql = llm_result.sql

        try:
            result.execution_rows = execute_sql(llm_result.sql)
        except Exception as exc:
            result.error = f"Execution error: {exc}"

    def _run_ast_guardcache(
        self, result: PipelineResult, question: str, tenant_id: int
    ) -> None:
        """Full pipeline: cache → RAG → LLM → AST → cache store → execute."""
        # 1. Cache lookup
        cached_sql, hit = self.cache.lookup(question, tenant_id)
        result.cache_hit = hit

        if hit and cached_sql:
            result.generated_sql = cached_sql
            result.verified_sql = cached_sql
            try:
                result.execution_rows = execute_sql(cached_sql)
            except Exception as exc:
                result.error = f"Execution error on cached SQL: {exc}"
            return

        # 2. Hybrid RAG retrieval
        table_names, ddl_strings = self.retriever.retrieve(question)
        schema_ddl = "\n".join(ddl_strings)
        result.pruned_tables_count = len(table_names)
        result.pruned_ddl = schema_ddl

        # 3. LLM generation with AST self-correction
        llm_result = self._call_llm(question, schema_ddl, tenant_id, verify=True)
        result.generated_sql = llm_result.sql
        result.ast_retries = llm_result.retries_used

        if llm_result.error:
            result.error = llm_result.error
            result.verified_sql = llm_result.sql
            return

        result.verified_sql = llm_result.sql

        # 4. Store in cache
        self.cache.store(question, llm_result.sql, tenant_id)

        # 5. Execute
        try:
            result.execution_rows = execute_sql(llm_result.sql)
        except Exception as exc:
            result.error = f"Execution error: {exc}"

    # ----- Helpers ----------------------------------------------------------

    def _call_llm(
        self,
        question: str,
        schema_ddl: str,
        tenant_id: int,
        verify: bool = False,
    ) -> LLMResult:
        """Dispatch to mock or real LLM."""
        verify_fn = verify_and_inject_ast if verify else None

        if self.use_mock_llm:
            return mock_generate_sql(
                question=question,
                schema_ddl=schema_ddl,
                verify_fn=verify_fn,
                tenant_id=tenant_id,
            )
        return generate_sql(
            question=question,
            schema_ddl=schema_ddl,
            verify_fn=verify_fn,
            tenant_id=tenant_id,
        )
