#!/bin/bash

# TwoTowerMLRetrieval Deployment Script
# This script ensures proper deployment with all required artifacts

set -e  # Exit on any error

echo "🚀 Starting TwoTowerMLRetrieval Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_status "Docker is available"

# Validate deployment artifacts
echo "🔧 Validating deployment prerequisites..."
python3 deployment_setup.py

if [ $? -ne 0 ]; then
    print_error "Deployment validation failed. Please fix the issues above."
    exit 1
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t twotower-retrieval:latest .

if [ $? -ne 0 ]; then
    print_error "Docker build failed"
    exit 1
fi

print_status "Docker image built successfully"

# Stop existing container if running
if [ "$(docker ps -q -f name=twotower-app)" ]; then
    echo "🛑 Stopping existing container..."
    docker stop twotower-app
    docker rm twotower-app
fi

# Deploy using Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "🚀 Deploying with Docker Compose..."
    docker-compose up -d
    print_status "Application deployed with Docker Compose"
else
    # Fallback to regular Docker run
    echo "🚀 Deploying with Docker run..."
    docker run -d \
        --name twotower-app \
        -p 8888:8888 \
        -v "$(pwd)/artifacts:/app/artifacts:ro" \
        -v "$(pwd)/frontend/chroma_store:/app/frontend/chroma_store:ro" \
        --restart unless-stopped \
        twotower-retrieval:latest
    print_status "Application deployed with Docker run"
fi

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Health check
MAX_RETRIES=12
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8888/ > /dev/null 2>&1; then
        print_status "Application is healthy and running!"
        echo "🌐 Access your application at: http://localhost:8888"
        exit 0
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "⏳ Attempt $RETRY_COUNT/$MAX_RETRIES - waiting for application..."
    sleep 5
done

print_error "Application failed to start properly"
echo "📋 Check logs with: docker logs twotower-app"
exit 1 