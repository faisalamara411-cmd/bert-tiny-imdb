# Backend Dockerfile for FastAPI
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn pydantic requests

# Copy backend code and trained model
COPY backend_app.py .
COPY bert-tiny-imdb ./bert-tiny-imdb

# Expose FastAPI port
EXPOSE 8000

# Run backend
CMD ["uvicorn", "backend_app:app", "--host", "0.0.0.0", "--port", "8000"]













