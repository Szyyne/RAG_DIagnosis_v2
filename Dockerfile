# HuggingFace Spaces — RAG Diagnostic Gym
# Runtime: Docker SDK  |  Port: 7860 (required by HF Spaces)

FROM python:3.11-slim

# HF Spaces requirements
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git && \
    rm -rf /var/lib/apt/lists/*

# Python deps — split for better layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "openenv-core>=0.2.2" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "websockets>=12.0" \
        "pydantic>=2.6.0" \
        "pyyaml>=6.0.1" \
        "gradio>=4.25.0"

# Copy source
COPY rag_diagnostic_gym/ ./rag_diagnostic_gym/
COPY agents/             ./agents/
COPY openenv.yaml        .
COPY app.py              .

# Install package in editable mode
RUN pip install --no-cache-dir -e . --no-deps

# HF Spaces runs as non-root
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]
