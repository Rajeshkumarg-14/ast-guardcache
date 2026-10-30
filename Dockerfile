# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# AST-GuardCache — Production Dockerfile
# Uses uv for dependency management on the official astral image.
# Pre-downloads the sentence-transformer model at build time.
# ---------------------------------------------------------------------------

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (cache layer)
COPY pyproject.toml uv.lock* ./

# Install production dependencies
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# Copy application code
COPY app/ app/
COPY scripts/ scripts/
COPY tests/ tests/
COPY .env.example .env.example

# Pre-download the embedding model at build time
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose FastAPI port
EXPOSE 8000

# Run
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
