from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os
import re

app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in Render environment

class HackRxRequest(BaseModel):
    documents: str  # URL to the PDF file
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# -- Step 1: Download and extract PDF text from URL --
def extract_text_from_pdf_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

# -- Step 2: Clean whitespace and line breaks --
def clean_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple whitespace
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
    return text.strip()

# -- Step 3: Remove repeated headers/footers (appearing 5+ times) --
def remove_repeated_lines(text: str, threshold=5):
    lines = text.split("\n")
    line_counts = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1
    filtered = [line for line in lines if line_counts[line] < threshold]
    return "\n".join(filtered)

# -- Step 4: Truncate to Gemini input size limit (20,000 chars) --
def truncate_context(text: str, max_chars=20000):
    return text[:max_chars]

# -- Step 5: Call Gemini API --
def ask_gemini(question: str, context: str) -> str:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
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

# -- API Route --
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        full_text = extract_text_from_pdf_url(data.documents)
        full_text = clean_whitespace(full_text)
        full_text = remove_repeated_lines(full_text)
        full_text = truncate_context(full_text)
    except Exception as e:
        return {"answers": [f"Failed to read PDF: {e}"] * len(data.questions)}
    
    answers = [ask_gemini(q, full_text) for q in data.questions]
    return {"answers": answers}
