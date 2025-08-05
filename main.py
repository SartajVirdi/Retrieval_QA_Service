from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os
import tiktoken
import numpy as np
import faiss

app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Add this to your environment

# Tokenizer setup
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text):
    return len(tokenizer.encode(text))

def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_token_len = num_tokens(word + " ")
        if current_tokens + word_token_len <= max_tokens:
            current_chunk.append(word)
            current_tokens += word_token_len
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Request/Response schema
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# Extract PDF text from URL
def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    doc = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# Embed using Gemini embedding model
def embed_texts(texts: List[str]) -> np.ndarray:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    vectors = []
    for text in texts:
        body = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        res = requests.post(endpoint, headers=headers, json=body)
        res.raise_for_status()
        vectors.append(res.json()['embedding']['value'])
    return np.array(vectors).astype("float32")

# Gemini answer generator
def ask_gemini(context: str, question: str) -> str:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    prompt = f"""Answer the following question based only on the given context.\n\nContext:\n\"\"\"\n{context}\n\"\"\"\n\nQuestion: {question}\nAnswer:"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        res = requests.post(endpoint, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"Gemini error: {e}"

@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        full_text = extract_text_from_pdf_url(data.documents)
        chunks = chunk_text(full_text, max_tokens=500)
        embeddings = embed_texts(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
    except Exception as e:
        return {"answers": [f"Error processing document: {e}"] * len(data.questions)}

    answers = []
    for question in data.questions:
        q_embedding = embed_texts([question])
        _, I = index.search(q_embedding, k=3)
        top_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join(top_chunks)
        answers.append(ask_gemini(context, question))
    return {"answers": answers}
