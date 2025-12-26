## Multi-stage Dockerfile for the project
## Stages: base -> builder -> runtime

### Base stage: minimal Python runtime and defaults
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app


### Builder stage: install build tools, uv and project dependencies
FROM base AS builder

# avoid interactive prompts from debian packages
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \ 
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency manifests first to leverage layer caching
COPY pyproject.toml uv.lock /app/

# Install uv and sync dependencies. uv will create/install into an environment
# We then copy the resolved site-packages into /usr/local so the runtime stage
# can reuse the installed packages without carrying build tools.
RUN pip install --upgrade pip --root-user-action=ignore\
    && pip install uv --root-user-action=ignore\
    && uv sync --no-dev

# Determine where site-packages were installed and copy them to a stable location
RUN python - <<'PY'
import sys, site, shutil, os

# Determine destination
dst = '/usr/local/lib/python{major}.{minor}/site-packages'.format(major=sys.version_info.major, minor=sys.version_info.minor)
os.makedirs(dst, exist_ok=True)

# Get absolute path of destination to compare later
dst_abs = os.path.abspath(dst)

# Copy all site-packages
for src in site.getsitepackages():
    if os.path.isdir(src):
        src_abs = os.path.abspath(src)
        # ONLY copy if the source is not the same as the destination
        if src_abs != dst_abs:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied from {src} to {dst}")
        else:
            print(f"Skipping {src} (same as destination)")

print('Consolidation complete')
PY

### Runtime stage: copy only installed packages and application source
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy preinstalled packages from builder into system site-packages
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy preinstalled packages from builder into system site-packages
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# Copy only the application code needed at runtime
COPY api/ /app/api/
COPY mylib/ /app/mylib/
COPY model/ /app/model/
COPY templates/ /app/templates/
COPY pyproject.toml uv.lock /app/

# copy uv CLI from builder
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

EXPOSE 8000

# Default command: run the FastAPI app with uvicorn
CMD ["uv", "run", "--no-dev", "uvicorn", "api.fastapi_main:app", "--host", "0.0.0.0", "--port", "8000"]