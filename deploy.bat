@echo off
title 🚀 BH-EA Cloud Run Deployment
color 1F
echo.
echo ==========================================================
echo   Borehole Exploration / Surveying Analytics (BH-EA)
echo   Automated Deployment to Google Cloud Run
echo ==========================================================
echo.

REM --- Step 1: Git sync ---
echo 🔄 Pulling latest changes from GitHub...
git pull origin main
if %errorlevel% neq 0 (
    echo ❌ Git pull failed. Check your internet or repo access.
    pause
    exit /b 1
)

REM --- Step 2: Build container ---
echo 🏗️ Building container image for Cloud Run...
gcloud builds submit --tag gcr.io/practical-day-179721/bh-ea-dashboard .
if %errorlevel% neq 0 (
    echo ❌ Build failed. Check Dockerfile or gcloud auth.
    pause
    exit /b 1
)

REM --- Step 3: Deploy to Cloud Run ---
echo 🚀 Deploying service [bh-ea-dashboard]...
gcloud run deploy bh-ea-dashboard ^
  --image gcr.io/practical-day-179721/bh-ea-dashboard:latest ^
  --region africa-south1 ^
  --allow-unauthenticated
if %errorlevel% neq 0 (
    echo ❌ Deployment failed.
    pause
    exit /b 1
)

echo ✅ Deployment successful!
echo 🌍 Open your dashboard:
echo https://bh-ea-dashboard-1097427212316.africa-south1.run.app
echo.
pause
