#!/bin/bash
set -e
echo "🔄 Pulling latest changes from GitHub..."
git pull origin main

echo "🏗️ Building new container image..."
gcloud builds submit --tag gcr.io/practical-day-179721/bh-ea-dashboard .

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy bh-ea-dashboard \
  --image gcr.io/practical-day-179721/bh-ea-dashboard:latest \
  --region africa-south1 \
  --allow-unauthenticated

echo "✅ Deployment complete!"
