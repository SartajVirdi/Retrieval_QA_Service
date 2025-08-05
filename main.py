# main.py
import fitz  # PyMuPDF
import requests
import shutil
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Claude API ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
LLM_MODEL = "anthropic/claude-3-sonnet"

def ask_llm(question, context):
    prompt = f"""
You are an insurance policy assistant. Based on the context below, answer the question precisely.

Context:
{context}

Question:
{question}

Answer:
"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"].strip()

# --- PDF Extraction ---
def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

# --- Embedding & FAISS ---
def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def build_faiss(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, chunks

def search_chunks(index, query, chunks, k=5):
    q_embed = model.encode([query])
    _, indices = index.search(np.array(q_embed), k)
    return [chunks[i] for i in indices[0]]

# --- FastAPI ---
class RequestBody(BaseModel):
    documents: str
    questions: list

@app.post("/hackrx/run")
async def run(payload: RequestBody):
    # Download PDF
    pdf_url = payload.documents
    response = requests.get(pdf_url, stream=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(response.raw, tmp_file)
        path = tmp_file.name

    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    index, chunk_data = build_faiss(chunks)

    answers = []
    for question in payload.questions:
        top_chunks = search_chunks(index, question, chunk_data)
        context = "\n".join(top_chunks)
        answer = ask_llm(question, context)
        answers.append(answer)

    return { "answers": answers }
