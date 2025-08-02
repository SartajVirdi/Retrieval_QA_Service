from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import base64
import os

app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in Render environment

class HackRxRequest(BaseModel):
    documents: str  # URL to the PDF file
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# -- Util: Download and extract PDF text from URL --
def extract_text_from_pdf_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

# -- Util: Call Gemini with question + context --
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
    except Exception as e:
        return {"answers": [f"Failed to read PDF: {e}"] * len(data.questions)}
    
    answers = [ask_gemini(q, full_text) for q in data.questions]
    return {"answers": answers}
