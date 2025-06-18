# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be required by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy the requirements files and install dependencies
COPY requirements.txt .
COPY frontend/requirements.txt frontend_requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r frontend_requirements.txt

# Copy the application code into the container
COPY ./frontend /app/frontend
COPY ./backend /app/backend
COPY ./artifacts /app/artifacts
COPY ./data /app/data

# Create directory for ChromaDB
RUN mkdir -p /app/chroma_store

# Expose the port the app runs on
EXPOSE 8888

# Change to frontend directory and run the FastAPI app
WORKDIR /app/frontend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"] 