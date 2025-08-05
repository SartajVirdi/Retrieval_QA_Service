from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os
from openai import OpenAI

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Extract text from PDF ---
def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    doc = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# --- Ask Gemini 2.5 Pro via OpenRouter ---
def ask_openrouter(question: str, context: str) -> str:
    prompt = f"Answer the following question based on the document context:\n\nContext:\n{context[:20000]}\n\nQuestion: {question}"
    try:
        completion = client.chat.completions.create(
            model="google/gemini-2.5-pro-exp-03-25",
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": "https://yourdomain.com",  # optional
                "X-Title": "InsuraBot",  # optional
            },
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Gemini error: {e}"

# --- Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        context = extract_text_from_pdf_url(data.documents)
    except Exception as e:
        return {"answers": [f"PDF error: {e}"] * len(data.questions)}
    
    answers = [ask_openrouter(q, context) for q in data.questions]
    return {"answers": answers}
