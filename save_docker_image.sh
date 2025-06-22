#!/bin/bash

echo "💾 Docker Image Save & Export Utility"
echo "===================================="

# Build the image if it doesn't exist
echo "🔨 Building Docker image..."
docker build -t twotower-app:latest .

# Save image to tar file (for backup/transfer)
echo "💾 Saving image to tar file..."
docker save -o twotower-app.tar twotower-app:latest
echo "✅ Image saved as 'twotower-app.tar' ($(du -h twotower-app.tar | cut -f1))"

# Compress the tar file
echo "🗜️  Compressing image..."
gzip twotower-app.tar
echo "✅ Compressed image saved as 'twotower-app.tar.gz' ($(du -h twotower-app.tar.gz | cut -f1))"

# Show image details
echo ""
echo "📊 Image Details:"
docker images twotower-app:latest

echo ""
echo "🚀 Deployment Options:"
echo "1. Local: docker load < twotower-app.tar.gz && docker run -p 8888:8888 twotower-app"
echo "2. Cloud: Push to container registry (see DEPLOYMENT_GUIDE.md)"
echo "3. Transfer: Copy twotower-app.tar.gz to another machine"

echo ""
echo "🏷️  Tagging for registries:"
echo "Docker Hub: docker tag twotower-app:latest yourusername/twotower-app:latest"
echo "Google: docker tag twotower-app:latest gcr.io/project-id/twotower-app:latest"
echo "AWS ECR: docker tag twotower-app:latest account.dkr.ecr.region.amazonaws.com/twotower-app:latest" 