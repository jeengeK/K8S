# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (optimization for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables (will be overridden by k8s config)
ENV PYTHONUNBUFFERED=1 \
    BUCKET_NAME="" \
    SQS_URL="" \
    TELEGRAM_TOKEN="" \
    POLYBOT_RESULTS_URL=""

# Run the bot
CMD ["python", "bot.py"]