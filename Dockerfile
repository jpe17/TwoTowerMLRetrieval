# Use official Python image
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary backend files
COPY backend/ ./backend/

# Copy necessary frontend files
COPY frontend/main.py ./frontend/
COPY frontend/index.html ./frontend/
COPY frontend/config.json ./frontend/
COPY frontend/chroma_store/ ./frontend/chroma_store/

# Copy the specific artifacts run (as specified in backend config)
COPY artifacts/run-20250619_232020/ ./artifacts/run-20250619_232020/

# Copy essential data files
COPY data/embeddings.npy ./data/
COPY data/word_to_idx.pkl ./data/

# Expose port
EXPOSE 8888

# Start FastAPI app
CMD ["uvicorn", "frontend.main:app", "--host", "0.0.0.0", "--port", "8888"]