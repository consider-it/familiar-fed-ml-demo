# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update -y && apt upgrade -y

# Copy requirements file
COPY requirements.txt .

RUN apt-get install -y libgl1 python3-pip libglib2.0-0

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

RUN pip install dist/cit_fl_participation-0.0.9-py3-none-any.whl

# Expose port for coordinator service
EXPOSE 5051

# Set environment variable for coordinator URL with default value
ENV COORDINATOR_URL=127.0.0.1:5051

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Run the application
ENTRYPOINT ["python", "participate.py"]
