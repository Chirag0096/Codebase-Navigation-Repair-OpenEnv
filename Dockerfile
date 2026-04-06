FROM python:3.11-slim

# Create non-root user for security — MANDATORY for running agent code safely
RUN useradd -m -u 1000 envuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Make repo_templates readable
RUN chmod -R 755 repo_templates/

# Create temp directory for working copies
RUN mkdir -p /tmp/openenv_work && chmod 777 /tmp/openenv_work

# Switch to non-root for security
USER envuser

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
