from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os
from openai import OpenAI
import faiss
import numpy as np
import tiktoken

app = FastAPI()

# Environment variable (or paste your key for local testing)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.5-flash-lite"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Tokenizer setup for token-based chunking
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text):
    return len(tokenizer.encode(text))

def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        token_len = num_tokens(word + " ")
        if current_tokens + token_len <= max_tokens:
            current_chunk.append(word)
            current_tokens += token_len
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    responses = []
    for text in texts:
        res = client.embeddings.create(
            model="openai/text-embedding-ada-002",
            input=text,
        )
        responses.append(res.data[0].embedding)
    return np.array(responses).astype("float32")

def build_faiss_index(chunks: List[str]):
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def get_top_k_chunks(question, chunks, index, embeddings, k=3):
    q_embedding = embed_texts([question])
    _, I = index.search(np.array(q_embedding).astype("float32"), k)
    return [chunks[i] for i in I[0]]

def call_gemini(question: str, context: str) -> str:
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://your-site.com",  # Optional
                "X-Title": "GeminiDocBot",  # Optional
            },
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer the following question based only on the given context. Provide a clear, complete answer.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""
                }
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Gemini error: {e}"

# Request/Response Models
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# PDF extractor
def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    doc = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        raw_text = extract_text_from_pdf_url(data.documents)
        chunks = chunk_text(raw_text, max_tokens=500)
        index, _ = build_faiss_index(chunks)
    except Exception as e:
        return {"answers": [f"PDF processing error: {e}"] * len(data.questions)}

    answers = []
    for question in data.questions:
        top_chunks = get_top_k_chunks(question, chunks, index, _, k=3)
        combined_context = "\n\n".join(top_chunks)
        answer = call_gemini(question, combined_context)
        answers.append(answer)

    return {"answers": answers}
