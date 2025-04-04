FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# [Optional] Download YOLOv5 weights during build if internet access is a concern
# RUN python -c "import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt', '/app/yolov5s.pt')"

# Set environment variables (overridden during runtime or Kubernetes deployment)
ENV AWS_REGION="eu-north-1"
ENV SQS_QUEUE_URL="your_sqs_queue_url"
ENV S3_BUCKET_NAME="your_s3_bucket_name"
ENV MONGO_CONNECTION_STRING=""
# No default mongo connection string.
ENV POLYBOT_URL="http://svc-polybot:8443/results"
# [Optional] If you downloaded weights during build
# ENV MODEL_PATH="/app/yolov5s.pt"

# Expose port if your app is a web service (optional)
EXPOSE 5000

CMD ["python", "app.py"]