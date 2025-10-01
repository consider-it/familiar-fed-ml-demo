# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for coordinator service
EXPOSE 5051

# Set environment variable for coordinator URL with default value
ENV COORDINATOR_URL=127.0.0.1:5051

# Run the application
ENTRYPOINT ["python", "participate.py"]