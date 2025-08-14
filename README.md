# Retrieval_QA_Service
<a href="https://www.koyeb.com/">
  <img src="assets/logo.png" alt="Retrieval QA Service Logo" align="right" width="120"/>
</a>

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&repository=SartajVirdi/Retrieval_QA_Service)

> ⚠️ After deployment, make sure to add your `OPENAI_API_KEY` in the environment variables section of the Koyeb dashboard.

A FastAPI-based retrieval-augmented question answering API.  
It:
- Downloads and extracts text from a PDF URL
- Chunks text and builds embeddings with [Sentence Transformers](https://www.sbert.net/)
- Indexes with [FAISS](https://faiss.ai/) for similarity search
- Sends top-k context to an LLM via [OpenAI API key](https://platform.openai.com/account/api-keys)

---

## Features
- **PDF ingestion** from any URL (with size/page limits for safety)
- **Chunking with overlap** for better context retention
- **Cosine similarity search** using FAISS
- **Multiple chunk context** for improved LLM answers
- **Singleton model loading** for speed
- **OPENAI API integration** 
- **CORS support** for browser-based clients
- **Docker-ready** (deploys cleanly on Koyeb)

---

## Requirements

- Python **3.11**
- OpenAI API key (https://platform.openai.com/account/api-keys)

---

## Installation (Local)

```bash
# 1) Clone
git clone https://github.com/SartajVirdi/Retrieval_QA_Service.git
cd Retrieval_QA_Service

# 2) Create venv
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 4) Set env
export OPENAI_API_KEY="your_key_here"
# Optional:
# export OPENAI_MODEL="gpt-4o-mini"
# export TOP_K=3

# 5) Run locally (entry file is main.py)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```
---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```
Retrieval QA Endpoint
POST /hackrx/run
```
{
  "documents": "https://example.com/mydoc.pdf",
  "questions": [
    "What is the policy on data retention?",
    "Who is the responsible authority?"
  ],
  "top_k": 3,
  "max_pages": 50
}
```
Example Response:
```
{
  "answers": {
    "What is the policy on data retention?": {
      "answer": "Data retention is limited to 3 years unless legally required otherwise.",
      "evidence": [
        {
          "text": "Data retention shall not exceed three years...",
          "score": 0.9123,
          "index": 5
        }
      ]
    }
  }
}
```
## Docker Deployment
Build:
```
docker build -t hackrx-qa .
```
Run:
```
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your_key_here" \
  hackrx-qa
```
API will be available at:
```
http://localhost:8000
```

## Deploying on Koyeb
This repo includes a Dockerfile and .dockerignore. No code changes needed.
- Create the service
- Push to GitHub: SartajVirdi/Retrieval_QA_Service.
- In Koyeb: Create → App → Deploy Service.
- Source: GitHub → select this repo → build type Dockerfile.
- Start command (use the Dockerfile default or set explicitly):
```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker -w ${WORKERS:-2} -b 0.0.0.0:${PORT} --timeout ${TIMEOUT:-120}
```
## Environment variables:
- OPENAI_API_KEY = your OpenAI key
- ENABLE_CORS=true, MAX_PDF_MB=20, MAX_PDF_PAGES=200,
- CHUNK_CHAR_TARGET=1200, CHUNK_CHAR_OVERLAP=180.
## Ports:
- Port: 8000 (locally)
- Protocol: HTTP
- Path: /health
- Public: ON
- On Koyeb, the PORT environment variable is set automatically — the Dockerfile is configured to use it.
## Verify
Replace <your-app>.koyeb.app with your domain:
```bash
curl https://<your-app>.koyeb.app/health

curl -X POST "https://<your-app>.koyeb.app/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
        "documents": "https://arxiv.org/pdf/1706.03762.pdf",
        "questions": ["What problem does the paper address?"],
        "top_k": 3,
        "max_pages": 5
      }'
```
## Tips:
- Instance size / workers: If you hit OOM, set WORKERS=1 or choose a larger instance.
- Auto-deploy: Enable “Auto deploy on push” in Koyeb service settings.
- Custom domain: Add your domain in Koyeb → Domains and map it to the service.
- Logs: Use the Logs tab to view Gunicorn/Uvicorn output.
- CORS: Keep ENABLE_CORS=true for browser clients; restrict origins in code for production.
## Environment Variables

| Variable             | Required | Default | Description                          |
|----------------------|----------|---------|--------------------------------------|
| `OPENAI_API_KEY`     | Yes      | —       | OpenAI API key                       |
| `TOP_K`              | No       | `3`     | Top chunks to provide as context     |
| `MAX_PDF_MB`         | No       | `20`    | Max PDF size (MB)                    |
| `MAX_PDF_PAGES`      | No       | `200`   | Max pages to extract                 |
| `CHUNK_CHAR_TARGET`  | No       | `1200`  | Target characters per chunk          |
| `CHUNK_CHAR_OVERLAP` | No       | `180`   | Characters overlapped between chunks |
| `ENABLE_CORS`        | No       | `true`  | Enable CORS for browsers             |
| `WORKERS`            | No       | `2`     | Gunicorn workers                     |
| `TIMEOUT`            | No       | `120`   | Gunicorn worker timeout (seconds)    |

> **Note:** This app uses the [OpenAI API](https://platform.openai.com) for LLM calls.
> Set the environment variable `OPENAI_API_KEY` in your deployment settings or `.env` file.  
> Ensure your code calls `call_openai()` when making API requests.

## Troubleshooting

**ModuleNotFoundError: app**  
- Your entry file is `main.py`. Ensure the start command targets `main:app` (see Koyeb section).

**401/403 from OpenAI**  
- Check `OPENAI_API_KEY` in Koyeb → Environment.

**PDF too large / slow**  
- Adjust `MAX_PDF_MB` / `MAX_PDF_PAGES`, or test with a smaller document.

**CORS issues in browser**  
- Keep `ENABLE_CORS=true` or proxy via your frontend; for production, restrict allowed origins in code.


