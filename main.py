import logging
import os
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# Enable logging
logging.basicConfig(level=logging.INFO)
logging.info("🚀 App is starting...")

# Get Claude API key
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logging.error("❌ OPENROUTER_API_KEY not found in environment!")
    raise ValueError("Missing Claude API key")
else:
    logging.info("✅ Claude API key loaded.")

# FastAPI app
app = FastAPI()

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": str(exc)})

# Health check route
@app.get("/")
async def root():
    return {"status": "🟢 API is running!"}


# PDF Q&A input model
class QARequest(BaseModel):
    documents: str  # PDF URL
    questions: list[str]


# Utility: Download and extract PDF
def extract_text_from_pdf_url(pdf_url: str) -> str:
    logging.info(f"📄 Downloading PDF: {pdf_url}")
    response = requests.get(pdf_url)
    response.raise_for_status()
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    doc = fitz.open("temp.pdf")
    text = "\n".join([page.get_text() for page in doc])
    logging.info("✅ PDF text extracted.")
    return text


# Utility: Chunk text into embeddings
def embed_and_index(text: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("📦 Loaded sentence-transformers model.")

    paragraphs = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    logging.info(f"🧩 Total chunks: {len(paragraphs)}")

    embeddings = model.encode(paragraphs)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, paragraphs, model


# Route: /hackrx/run
@app.post("/hackrx/run")
async def hackrx_query(payload: QARequest):
    try:
        text = extract_text_from_pdf_url(payload.documents)
        index, embeddings, chunks, model = embed_and_index(text)

        final_answers = {}
        for q in payload.questions:
            q_embedding = model.encode([q])
            D, I = index.search(np.array(q_embedding), k=1)
            matched_chunk = chunks[I[0][0]]

            logging.info(f"🔍 Top match for question '{q}': {matched_chunk[:100]}...")

            # Ask Claude
            prompt = f"""You are an expert policy assistant. Based on the below document chunk, answer this question:\n\nChunk: \"{matched_chunk}\"\n\nQuestion: \"{q}\"\nAnswer:"""
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://chat.openai.com",
                "X-Title": "hackrx-submission"
            }
            data = {
                "model": "anthropic/claude-3-sonnet",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
            logging.info(f"📨 Claude response code: {response.status_code}")
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"]
            final_answers[q] = {
                "answer": answer,
                "matched_chunk": matched_chunk
            }

        return {"answers": final_answers}

    except Exception as e:
        logging.error(f"❌ Error during processing: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
