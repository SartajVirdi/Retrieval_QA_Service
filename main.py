from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in environment (Render/ngrok)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class HackRxRequest(BaseModel):
    documents: str  # URL to the PDF
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Extract plain text from PDF URL ---
def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    doc = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# --- Split text into chunks ---
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --- Embed chunks using SentenceTransformer ---
def embed_chunks(chunks: List[str]) -> np.ndarray:
    return np.array(embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False))

# --- FAISS index to retrieve top-k chunks ---
def get_top_k_chunks(query: str, chunks: List[str], chunk_embeddings: np.ndarray, k: int = 5) -> List[str]:
    query_embedding = embedder.encode(query, convert_to_numpy=True)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]

# --- Ask Gemini API ---
def ask_gemini(question: str, context: str) -> str:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    prompt = f"Answer the following question based on the document context:\n\nContext:\n{context}\n\nQuestion: {question}"
    payload = {
        "contents": [
            {
                "parts": [ {"text": prompt} ]
            }
        ]
    }
    try:
        res = requests.post(endpoint, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"Gemini error: {e}"

# --- API Route ---
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        text = extract_text_from_pdf_url(data.documents)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
    except Exception as e:
        return {"answers": [f"PDF/Embedding error: {e}"] * len(data.questions)}

    answers = []
    for question in data.questions:
        top_chunks = get_top_k_chunks(question, chunks, embeddings)
        context = "\n".join(top_chunks)
        answer = ask_gemini(question, context)
        answers.append(answer)

    return {"answers": answers}
