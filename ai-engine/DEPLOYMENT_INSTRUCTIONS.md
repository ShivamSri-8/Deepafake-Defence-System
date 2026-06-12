# EDDS AI Engine Deployment Guide

This guide covers deploying the production-ready AI Engine to [Railway.app](https://railway.app/), a platform well-suited for machine learning inference workloads.

## Why Railway?
Render's free tier is strictly limited to 512MB of RAM. The AI Engine, even with a single PyTorch EfficientNet model and batching enabled, may consume ~600-800MB during video frame extraction and inference. Railway provides 500 hours of usage and 2GB RAM on the free tier, making it the ideal choice to prevent Out-Of-Memory (OOM) errors.

## Prerequisites
1. A GitHub account.
2. A Railway account ([sign up here](https://railway.app/)).

## Deployment Steps

### 1. Push to GitHub
Ensure all your recent changes to the `ai-engine` folder are committed and pushed to your GitHub repository.

### 2. Create a Railway Project
1. Log in to your Railway dashboard.
2. Click **"New Project"**.
3. Select **"Deploy from GitHub repo"**.
4. Choose the `ShivamSri-8/Deepafake-Defence-System` repository.
5. Railway will detect the repository but won't know which folder to deploy initially.

### 3. Configure the Service
Once the service is created, click on it, go to the **Settings** tab, and configure the following:

#### Root Directory
- Under **Build**, find **Root Directory**.
- Set it to `/ai-engine`. This tells Railway to only deploy the AI backend, ignoring the frontend.

#### Environment Variables
Go to the **Variables** tab and add:
- `DEBUG=False`
- `USE_PYTORCH=True`
- `ALLOWED_ORIGINS=*` (or your specific frontend domain once deployed)
- `FAKE_THRESHOLD=0.60`
- `SUSPICIOUS_THRESHOLD=0.40`

### 4. Deploy
Railway will automatically trigger a build using the `Dockerfile` inside the `ai-engine` directory. 
- It will install the optimized `requirements.txt`.
- It will download PyTorch and load the `efficientnet_deepfake.pt` model.

### 5. Generate a Public URL
1. Go to the **Settings** tab of the service.
2. Scroll down to **Networking**.
3. Click **Generate Domain**.
4. Railway will provide a public URL (e.g., `https://edds-ai-engine-production.up.railway.app`).

### 6. Update the Main Backend
In your **Node.js Main Backend** repository (or environment variables), update the `AI_ENGINE_URL` to point to the new Railway domain you just generated.

## Verification
You can verify the deployment is working by visiting:
`https://<YOUR-RAILWAY-DOMAIN>/health`

It should return:
```json
{
  "status": "ok",
  "models_loaded": true,
  "framework": "pytorch",
  "active_models": [
    "EfficientNet-B4"
  ],
  "inference_ready": true
}
```
