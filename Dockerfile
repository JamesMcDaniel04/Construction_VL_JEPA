FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir .

# Copy source
COPY src/ src/
COPY configs/ configs/

# Install the package
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Training image
# ---------------------------------------------------------------------------
FROM base AS training
ENTRYPOINT ["pfil-train"]

# ---------------------------------------------------------------------------
# Inference image
# ---------------------------------------------------------------------------
FROM base AS inference
ENTRYPOINT ["pfil-infer"]

# ---------------------------------------------------------------------------
# API image
# ---------------------------------------------------------------------------
FROM base AS api
EXPOSE 8000
ENTRYPOINT ["pfil-api"]

# ---------------------------------------------------------------------------
# Ingestion image
# ---------------------------------------------------------------------------
FROM base AS ingestion
ENTRYPOINT ["pfil-ingest"]
