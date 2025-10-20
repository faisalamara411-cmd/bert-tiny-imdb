# Frontend Dockerfile for Streamlit
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install only needed packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit requests

# Copy frontend code
COPY frontend_app.py .

# Expose Streamlit port
EXPOSE 8501

# Run frontend
CMD ["streamlit", "run", "frontend_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

