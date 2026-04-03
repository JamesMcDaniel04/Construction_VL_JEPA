FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000

ENTRYPOINT ["mtc-api"]
