from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in Render environment

# Use lighter model for reduced memory footprint
embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

class HackRxRequest(BaseModel):
    documents: str  # URL to the PDF file
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# -- Preprocessing & Chunking --
def extract_text_from_pdf_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

def chunk_text(text, max_tokens=256):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def find_most_relevant_chunks(question, chunks, top_k=3):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return [chunks[i] for i in top_results]

def ask_gemini(question: str, context: str) -> str:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    prompt = f"Answer the following question based on the provided policy document context.\n\nContext:\n{context}\n\nQuestion: {question}"
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
        return f"Error from Gemini: {e}"

@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        full_text = extract_text_from_pdf_url(data.documents)
        chunks = chunk_text(full_text)
    except Exception as e:
        return {"answers": [f"Failed to process PDF: {e}"] * len(data.questions)}

    answers = []
    for q in data.questions:
        top_chunks = find_most_relevant_chunks(q, chunks)
        context = "\n".join(top_chunks)
        answers.append(ask_gemini(q, context))
    return {"answers": answers}
