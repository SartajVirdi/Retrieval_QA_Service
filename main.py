from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os

app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in Render dashboard

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

# --- Ask Gemini API ---
def ask_gemini(question: str, context: str) -> str:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    prompt = f"Answer the following question based on the document context:\n\nContext:\n{context[:20000]}\n\nQuestion: {question}"
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
        context = extract_text_from_pdf_url(data.documents)
    except Exception as e:
        return {"answers": [f"PDF error: {e}"] * len(data.questions)}
    
    answers = [ask_gemini(q, context) for q in data.questions]
    return {"answers": answers}
