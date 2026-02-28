FROM python:3.11-slim

# Install system dependencies for matplotlib and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run as non-root user for better security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Health-check: verify Gradio port is up (only relevant when app is running)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Keep container running (training / app are started manually via run.sh)
CMD ["tail", "-f", "/dev/null"]
