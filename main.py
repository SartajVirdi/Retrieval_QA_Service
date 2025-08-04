from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os
import re
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in Render environment

# Load embedding model for clause matching
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------- Request and Response Models -----------
class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# ----------- Step 1: Extract and Preprocess Text -----------
def extract_text_from_pdf_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        raw_text = "\n".join(page.get_text() for page in doc)
        return preprocess_text(raw_text)
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"(?<=\w)[.]{2,}", ".", text)  # clean up ellipsis
    return text.strip()

# ----------- Step 2: Clause Matching with Embeddings -----------
def match_clause(context, question):
    try:
        sentences = context.split(". ")
        question_embed = embedding_model.encode(question, convert_to_tensor=True)
        sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(question_embed, sentence_embeddings)[0]
        top_indices = similarities.topk(k=min(3, len(similarities))).indices
        matched = " ".join([sentences[i] for i in top_indices])
        return matched
    except:
        return context  # fallback to full context

# ----------- Step 3: Ask Gemini with Question + Matched Context -----------
def ask_gemini(question: str, context: str) -> str:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    prompt = f"Answer the following question based on the provided policy document context.\n\nContext:\n{context[:20000]}\n\nQuestion: {question}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"Error from Gemini: {e}"

# ----------- Step 4: API Endpoint -----------
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        full_text = extract_text_from_pdf_url(data.documents)
    except Exception as e:
        return {"answers": [f"Failed to read PDF: {e}"] * len(data.questions)}

    answers = []
    for question in data.questions:
        matched_context = match_clause(full_text, question)
        answer = ask_gemini(question, matched_context)
        answers.append(answer)

    return {"answers": answers}
