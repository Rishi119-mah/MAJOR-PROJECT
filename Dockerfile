# Production Dockerfile for Fake Review Detection API
FROM python:3.11-slim

WORKDIR /app

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories for models and optional mounted datasets
RUN mkdir -p /app/data /app/configs /app/outputs/models /app/outputs/experiments

# Copy application source code and documentation
COPY src/ /app/src/
COPY outputs/models/ /app/outputs/models/
COPY outputs/experiments/ /app/outputs/experiments/

# Expose FastAPI service port
EXPOSE 8000

# Healthcheck probe using Python urllib (zero curl dependency)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start production ASGI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
