# Deployment Guide

## Hugging Face Spaces (Docker)

1. Create a new Space at https://huggingface.co/spaces
2. Select **Docker** as the SDK
3. Push your repo to the Space (or link your GitHub)
4. Add `MISTRAL_API_KEY` as a secret in Space Settings
5. The app will auto-build and serve on the Space URL

The Dockerfile is already configured to expose port 8000.

## Render (Free Tier)

1. Connect your GitHub repo at https://render.com
2. Render auto-detects the Dockerfile
3. Add `MISTRAL_API_KEY` as an environment variable
4. Deploy — free tier gives you 750 hours/month

## Railway

1. Connect repo at https://railway.app
2. Auto-detects Dockerfile
3. Set `MISTRAL_API_KEY` in variables
4. Free tier: $5 credit/month

## Local Docker

```bash
docker build -t mindsphere-coach .
docker run -p 8000:8000 -e MISTRAL_API_KEY=your_key mindsphere-coach
```

Then open http://localhost:8000

## Without Docker

```bash
pip install -e .
python scripts/run_demo.py
```

## Note on API Key

The app works without a Mistral API key — it falls back to template-based responses. For the full LLM-powered experience, get a key at https://console.mistral.ai/
