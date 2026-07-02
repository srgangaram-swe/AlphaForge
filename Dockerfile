FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY alphaforge ./alphaforge
COPY scripts ./scripts
COPY apps ./apps
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e ".[dev]"

CMD ["python", "scripts/run_walk_forward.py", "--synthetic", "--fast"]
