# Reel2Real Personalised AI Core Engine

Dedicated microservice for Reel2Real (R2R) omnichannel social automation, powered by FastAPI, Google Gemini 1.5 Flash, and product search on Supabase Postgres (pgvector).

## Features
- **Dynamic Prompt Assembler**: Merchant Brand Persona (Tone, Language rules, Hinglish support, emoji density).
- **Post-to-Product Grounding**: Maps Instagram/Facebook posts to exact catalog products & SKUs.
- **Sentiment & Safety Guardrail**: Immediate de-escalation & human handoff when angry or refund keywords are detected.
- **Product Search**: Reads the backend's `Product` table in Supabase and keeps Gemini embeddings in `ai.product_embedding` (pgvector), refreshed when products change. Falls back to keyword search when embeddings are unavailable, and to an in-memory catalog when `DATABASE_URL` is empty.

## Quickstart
```bash
# 1. Activate environment
.\.venv\Scripts\activate

# 2. Set Gemini API Key in .env
cp .env.example .env

# 3. Run FastAPI server
uvicorn app.main:app --reload --port 8000
```
