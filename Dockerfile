# AlphaForge — research image with the C++ execution core built in.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY alphaforge ./alphaforge
COPY cpp ./cpp
COPY scripts ./scripts
COPY apps ./apps
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]" \
    && python scripts/build_native.py

# Default: run the no-network synthetic demo pipeline end to end.
CMD ["sh", "-c", "python scripts/run_walk_forward.py --synthetic && python scripts/run_backtest.py --latest && python scripts/generate_report.py"]
