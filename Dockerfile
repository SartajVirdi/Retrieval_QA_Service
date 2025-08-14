# Dockerfile
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# System deps: libgomp1 for faiss/torch; curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r app && useradd -m -g app app
WORKDIR /app

# Faster, reproducible installs
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface \
    HF_HOME=/home/app/.cache/huggingface

# Network
EXPOSE 8000

# Drop privileges
USER app

# Healthcheck (Render will hit /health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT:-8000}/health || exit 1

# Start server; Render sets $PORT. Tune via WORKERS/TIMEOUT envs if needed.
CMD ["sh", "-c", "gunicorn main:app -k uvicorn.workers.UvicornWorker -w ${WORKERS:-2} -b 0.0.0.0:${PORT:-8000} --timeout ${TIMEOUT:-120}"]
