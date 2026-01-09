FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock /app/
RUN pip install uv --root-user-action=ignore && uv sync --frozen --no-dev

FROM python:3.13-slim AS runtime
ENV PYTHONPATH="/app"
WORKDIR /app

# Copy ONLY the venv (contains all libs)
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy app code
COPY api/ /app/api/
COPY custom_lib/ /app/custom_lib/
COPY ml_modules/model/ /app/ml_modules/model/
COPY templates/ /app/templates/
COPY pyproject.toml uv.lock /app/
COPY metrics.json /app/metrics.json

# Copy uv binary
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

EXPOSE 8000

# Default command: run the FastAPI app with uvicorn
CMD ["uv", "run", "--no-dev", "uvicorn", "api.api_main:app", "--host", "0.0.0.0", "--port", "8000"]