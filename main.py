from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tiktoken
import json

app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Embedding model and tokenizer
model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = tiktoken.get_encoding("cl100k_base")

# ===================== Pydantic Models =====================
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxAnswer(BaseModel):
    question: str
    answer: str

class HackRxResponse(BaseModel):
    answers: List[HackRxAnswer]

# ===================== Utilities =====================
def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    pdf = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join([page.get_text() for page in pdf])

def chunk_text_tokenwise(text: str, max_tokens: int = 300) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

def build_faiss_index(chunks: List[str]):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def get_top_k_chunks(query: str, chunks: List[str], index, k: int = 5) -> str:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return "\n\n".join([chunks[i] for i in I[0]])

def ask_gemini_flash(question: str, context: str) -> dict:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    prompt = f"""You are an insurance policy assistant. 
Answer the user's question using only the provided context.
Provide the result in this JSON format:
{{
  "question": "...",
  "answer": "..."
}}

Context:
{context}

Question:
{question}
"""

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload)
        res.raise_for_status()
        text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
        return {
            "question": question,
            "answer": result.get("answer", text)
        }
    except Exception as e:
        return {
            "question": question,
            "answer": f"Gemini error: {e}"
        }

# ===================== Endpoint =====================
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        text = extract_text_from_pdf_url(data.documents)
        chunks = chunk_text_tokenwise(text, max_tokens=300)
        index, _ = build_faiss_index(chunks)
    except Exception as e:
        return {
            "answers": [{"question": q, "answer": f"PDF error: {e}"} for q in data.questions]
        }

    results = []
    for question in data.questions:
        context = get_top_k_chunks(question, chunks, index)
        results.append(ask_gemini_flash(question, context))

    return {"answers": results}
