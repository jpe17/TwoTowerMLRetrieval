#!/bin/bash

echo "🧪 Testing Docker Deployment for TwoTowerMLRetrieval"
echo "=================================================="

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t twotower-app .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully!"

# Run the container
echo "🚀 Starting container on port 8888..."
docker run -d -p 8888:8888 --name twotower-test twotower-app

# Wait a few seconds for startup
sleep 10

# Test if the service is responding
echo "🔍 Testing service health..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/)

if [ "$response" = "200" ]; then
    echo "✅ Service is healthy! Visit http://localhost:8888"
    echo "🎯 Test search endpoint..."
    curl -X POST "http://localhost:8888/search" \
         -H "Content-Type: application/json" \
         -d '{"query": "test query", "alpha": 0.5}'
else
    echo "❌ Service health check failed (HTTP $response)"
    echo "📋 Container logs:"
    docker logs twotower-test
fi

echo ""
echo "🧹 Cleanup (run 'docker rm -f twotower-test' to stop the test container)" 