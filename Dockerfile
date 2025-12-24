FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt


# Copy app
COPY . .

EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
