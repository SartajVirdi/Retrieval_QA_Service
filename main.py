from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uuid
import traceback

app = FastAPI()

# Load API Key
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing Claude API key")

# Load Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Request Format
class QARequest(BaseModel):
    documents: str
    questions: list

# Claude LLM Query
def ask_claude(messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourprojectname.com",  # optional
        "Content-Type": "application/json",
    }

    body = {
        "model": "anthropic/claude-3-sonnet",
        "messages": messages,
        "max_tokens": 1024,
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# PDF Text Extractor
def extract_text_from_pdf(url):
    local_path = f"/tmp/{uuid.uuid4().hex}.pdf"
    r = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(r.content)
    doc = fitz.open(local_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# Chunking and Embedding
def chunk_text(text, max_tokens=100):
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        if len((current + sentence).split()) > max_tokens:
            chunks.append(current.strip())
            current = sentence
        else:
            current += sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, chunks

def get_top_chunks(query, index, embeddings, chunks, top_k=5):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]

# FastAPI Endpoint
@app.post("/hackrx/run")
async def run_qa(req: QARequest):
    try:
        text = extract_text_from_pdf(req.documents)
        chunks = chunk_text(text)
        index, embeddings, raw_chunks = build_faiss_index(chunks)

        answers = []
        for question in req.questions:
            context_chunks = get_top_chunks(question, index, embeddings, raw_chunks)
            context = "\n\n".join(context_chunks)

            messages = [
                {"role": "system", "content": "You are a helpful assistant for answering questions based on the given policy document context."},
                {"role": "user", "content": f"Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}"}
            ]

            answer = ask_claude(messages)

            answers.append({
                "question": question,
                "answer": answer.strip(),
                "source_chunks": context_chunks
            })

        return {"answers": answers}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
