#!/bin/bash

# Render CLI Docker Service Creator
echo "🚀 Creating Render service from Docker image..."
echo "Image: docker.io/jpe1/twotower-retrieval:2.0"
echo ""

# Open Render dashboard to create service
echo "Opening Render dashboard..."
open "https://dashboard.render.com/select-repo?type=web"

echo ""
echo "📋 Instructions:"
echo "1. In the browser, click 'Deploy an existing image from a registry'"
echo "2. Enter image URL: docker.io/jpe1/twotower-retrieval:2.0"
echo "3. Click 'Next'"
echo "4. Name: two-tower-ml-retrieval"
echo "5. Region: Oregon (US West)"
echo "6. Instance Type: Free (for testing)"
echo "7. Click 'Create Web Service'"
echo ""
echo "✅ Your app will be live at: https://two-tower-ml-retrieval.onrender.com" 