# Kovai Finserv RAG API — production image.
#
# Two stages. The first one installs the dependencies with whatever compilers
# and headers pip needs; the second one copies the finished install tree and
# leaves all of that behind. The runtime image never contains a compiler.

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Only requirements.txt is copied here, so this layer is cached and the whole
# dependency install is skipped on every rebuild where the pins have not moved.
COPY requirements.txt .

# --prefix=/install puts the entire install under one directory that can be
# copied into the runtime stage in a single layer. --no-cache-dir stops pip
# from keeping a wheel cache we would otherwise have to delete afterwards.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# PYTHONUNBUFFERED is not a nicety. On a free cloud instance the platform's log
# viewer is usually the only debugging tool you have, and Python buffers stdout
# when it is not a terminal — so without this, the log line explaining a crash
# is still sitting in the buffer when the process dies, and you see nothing.
#
# HOME must be set before ingest runs, because that is where Chroma puts the
# downloaded ONNX embedding model.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser \
    PORT=8000

# A non-root user, and /app owned by it — the ingest step below runs as this
# user and has to be able to create the index directory inside /app.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app \
    && chown appuser:appuser /app

# The finished dependency tree from stage 1. No pip, no build tools, no cache.
COPY --from=builder /install /usr/local

WORKDIR /app

# Only what the API needs at runtime. ui/, eval/ and tests/ are deliberately
# absent: the UI deploys separately, and eval/ would drag PyTorch in.
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser scripts/ ./scripts/

USER appuser

# ===========================================================================
# THE MOST IMPORTANT LINE IN THIS FILE
# ===========================================================================
# This runs at BUILD time, and it runs as appuser. Two things happen here that
# must not happen at request time:
#
#   1. Chroma downloads its ~80 MB ONNX embedding model into
#      /home/appuser/.cache/chroma. Because this runs as appuser with HOME
#      already set, the cache lands in a path the running container can read.
#      Run it at startup instead and the first customer question waits on an
#      80 MB download.
#
#   2. The vector index is built at /app/chroma_db and baked into the image.
#      A free instance has an ephemeral filesystem — anything written at
#      runtime is gone on the next restart, so an index built at startup is
#      rebuilt on every single cold start. Baked into the image, it is just
#      there.
#
# If this line is removed, the API still starts and then fails on the first
# question with a missing-collection error.
RUN python scripts/ingest.py

EXPOSE 8000

# The health check uses python, not curl. python:3.12-slim has no curl, and
# installing one purely so a health check has something to call means shipping
# an entire extra package into every layer. The interpreter is already here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health')"

# Shell form on purpose — no JSON brackets. The exec form does not run a shell,
# so ${PORT} would be passed to uvicorn as the literal four characters "$PORT"
# and the container would never bind the port the platform assigns it.
#
# --workers 1 is a MEMORY calculation, not a CPU one. Each uvicorn worker is a
# full copy of the process, including its own loaded copy of the ONNX embedding
# model and its own Chroma client. Two workers on a 512 MB free instance is not
# "twice the throughput", it is an out-of-memory kill.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
