FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make the main script executable
RUN chmod +x main.py
RUN chmod +x app.py

# Create directory for models
RUN mkdir -p /models

# Set environment variables
ENV PYTHONPATH=/app
ENV OPENAI_API_BASE=http://localhost:8080/v1
ENV OPENAI_API_KEY=""

# Expose the API port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]