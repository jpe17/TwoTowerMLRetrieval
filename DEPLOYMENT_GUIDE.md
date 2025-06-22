# 🚀 TwoTower ML Retrieval - Deployment Guide

## Quick Start Testing

```bash
# Make the test script executable
chmod +x test_deployment.sh

# Run local test
./test_deployment.sh
```

## 📋 Deployment Options

### 1. **Local Development**
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or direct Docker
docker build -t twotower-app .
docker run -p 8888:8888 twotower-app

# Access at: http://localhost:8888
```

### 2. **Cloud Deployment Options**

#### **Option A: Render.com (Easiest)**
Your Dockerfile is already configured for Render! 

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Connect your GitHub repo
   - Create new "Web Service"
   - It will auto-detect your Dockerfile
   - Deploy! 🚀

#### **Option B: Railway.app**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

#### **Option C: Google Cloud Run**
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/twotower-app

# Deploy to Cloud Run
gcloud run deploy twotower-service \
  --image gcr.io/YOUR-PROJECT-ID/twotower-app \
  --platform managed \
  --port 8888 \
  --memory 2Gi
```

#### **Option D: AWS ECS/Fargate**
```bash
# Build and push to ECR
docker build -t twotower-app .
docker tag twotower-app:latest YOUR-ECR-URI/twotower-app:latest
docker push YOUR-ECR-URI/twotower-app:latest

# Deploy via ECS console or AWS CLI
```

#### **Option E: DigitalOcean App Platform**
```bash
# Create app.yaml
cat > .do/app.yaml << EOF
name: twotower-app
services:
- name: web
  source_dir: /
  dockerfile_path: Dockerfile
  build_command: ""
  http_port: 8888
  instance_count: 1
  instance_size_slug: basic-s
EOF

# Deploy via DigitalOcean console or doctl
```

### 3. **Production Considerations**

#### **Environment Variables**
```bash
# Add to your deployment environment
ENVIRONMENT=production
LOG_LEVEL=info
MAX_WORKERS=4
```

#### **Scaling Configuration**
```yaml
# For docker-compose.yml
deploy:
  replicas: 3
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

#### **Monitoring Setup**
```bash
# Add to requirements.txt for production monitoring
prometheus-client==0.16.0
structlog==23.1.0
```

## 🔒 Security Checklist

- [ ] Add HTTPS in production
- [ ] Set up proper CORS origins (not "*")
- [ ] Add rate limiting
- [ ] Monitor resource usage
- [ ] Set up log aggregation

## 🎯 Testing Your Deployment

### Health Check
```bash
curl -f http://your-domain.com/
```

### API Test
```bash
curl -X POST "http://your-domain.com/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "alpha": 0.7}'
```

### Load Test (optional)
```bash
# Install wrk
brew install wrk  # macOS
# or apt-get install wrk  # Ubuntu

# Run load test
wrk -t12 -c400 -d30s --script=test_search.lua http://your-domain.com/
```

## 📊 Performance Monitoring

Your app includes basic health checks. For production, consider:
- **Application Performance Monitoring**: New Relic, DataDog
- **Uptime Monitoring**: Pingdom, UptimeRobot
- **Log Analysis**: ELK Stack, Splunk

## 🔄 CI/CD Pipeline (Optional)

```yaml
# .github/workflows/deploy.yml
name: Deploy to Render
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Render
        run: echo "Render auto-deploys on git push"
```

## 🎉 Your Website is Ready!

Once deployed, you'll have:
- **🔍 Search Interface**: Beautiful web UI at your domain
- **🔌 API Endpoints**: RESTful search API
- **📊 Hybrid Search**: Semantic + keyword search
- **⚡ Fast Response**: Optimized ML inference
- **🔄 Auto-scaling**: Cloud platform handles traffic

Visit your deployed URL and you have a fully functional ML-powered search website! 🚀 