FROM python:3.8-slim-buster

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose necessary ports (if applicable)
EXPOSE 8443

# Set the entrypoint command
CMD ["python", "main.py"]