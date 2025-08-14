# Dockerfile
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# System deps: libgomp1 for faiss/sentence-transformers; curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r app && useradd -m -g app app
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface

# Network
EXPOSE 8000

# Drop privileges
USER app

# Healthcheck hits FastAPI /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-8000}/health || exit 1

# Start server (works on Koyeb/Render/local)
CMD ["sh", "-c", "gunicorn main:app -k uvicorn.workers.UvicornWorker -w ${WORKERS:-2} -b 0.0.0.0:${PORT:-8000} --timeout ${TIMEOUT:-120}"]
