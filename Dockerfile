# Use slim Python image to reduce size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 7860

# Start both FastAPI backend and Streamlit frontend
CMD uvicorn src.app:app --host 0.0.0.0 --port 8000 & streamlit run src/frontend.py --server.port 7860 --server.address 0.0.0.0










