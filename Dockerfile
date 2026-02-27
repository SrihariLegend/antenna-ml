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

# Keep container running
CMD ["tail", "-f", "/dev/null"]
