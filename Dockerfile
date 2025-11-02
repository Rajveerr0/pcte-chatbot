FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Set environment variable
ENV PORT=7860

# Start the application
CMD gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --timeout 120 --preload