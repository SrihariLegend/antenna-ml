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

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import sklearn, pandas, gradio; print('ok')" || exit 1

# Keep container running
CMD ["tail", "-f", "/dev/null"]
