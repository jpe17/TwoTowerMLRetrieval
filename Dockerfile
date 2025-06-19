# Multi-stage build for production deployment
FROM python:3.9-slim as builder

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the application code
COPY . .

# Validate deployment artifacts
RUN python deployment_setup.py

# Production stage
FROM python:3.9-slim as production

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy application code and built artifacts from builder stage
COPY --from=builder /app .

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8888/ || exit 1

# Run the uvicorn server when the container launches
CMD ["uvicorn", "frontend.main:app", "--host", "0.0.0.0", "--port", "8888"] 