# ── Stage 1: Builder — install heavyweight Python deps ───────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# System packages needed by some deps (pdfplumber → pdfminer; sentence-transformers → torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an isolated prefix so we can copy cleanly in the next stage
RUN pip install --no-cache-dir --prefix=/runtime -r requirements.txt


# ── Stage 2: Runtime — lean final image ──────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /runtime /usr/local

# Streamlit config: disable browser auto-launch & set server options
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    # Sentence-transformers / HuggingFace cache inside a named volume
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Copy application source
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
