# HackRx Retrieval QA API

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SartajVirdi/Retrieval_QA_Service)

A FastAPI-based retrieval-augmented question answering API.  
It:
- Downloads and extracts text from a PDF URL
- Chunks text and builds embeddings with [Sentence Transformers](https://www.sbert.net/)
- Indexes with [FAISS](https://faiss.ai/) for similarity search
- Sends top-k context to an LLM via [OpenRouter](https://openrouter.ai/)
- Returns answers with matched evidence chunks

---

## Features
- **PDF ingestion** from any URL (with size/page limits for safety)
- **Chunking with overlap** for better context retention
- **Cosine similarity search** using FAISS
- **Multiple chunk context** for improved LLM answers
- **Singleton model loading** for speed
- **OpenRouter API integration** (Claude, GPT, etc.)
- **CORS support** for browser-based clients
- **Docker-ready** for Render or any container hosting

---

## Requirements

- Python **3.11**
- [OpenRouter API key](https://openrouter.ai/keys)
- Internet access (to fetch PDFs and call OpenRouter)

---

## Installation (Local)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/hackrx-retrieval-qa.git
cd hackrx-retrieval-qa

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set environment variables
export OPENROUTER_API_KEY="your_key_here"
# Optional overrides:
# export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"
# export TOP_K=3

# 5. Run server locally
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

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
  -e OPENROUTER_API_KEY="your_key_here" \
  hackrx-qa
```
API will be available at:
```
http://localhost:8000
```
Set the environment variable:
```
OPENROUTER_API_KEY=your_key_here
```
