# -- coding: utf-8 --

from flask import Flask, request, jsonify
import openai
import fitz  # PyMuPDF
import numpy as np
import os
import re
import base64
import time
from io import BytesIO

app = Flask(__name__)

# ✅ Set OpenRouter API key and base
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# ✅ Decode base64 PDF and extract text
def extract_text_from_uploaded_pdf(b64_data):
    try:
        pdf_bytes = base64.b64decode(b64_data)
        doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"PDF processing error: {e}")

# ✅ Chunk text with overlap
def split_text(text, max_tokens=500, overlap=100):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = []
    tokens = 0
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if tokens + sentence_tokens <= max_tokens:
            chunk.append(sentence)
            tokens += sentence_tokens
        else:
            chunks.append(" ".join(chunk))
            chunk = chunk[-(overlap // 5):] + [sentence]
            tokens = sum(len(s.split()) for s in chunk)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    if not data or "query" not in data or "pdf_base64" not in data:
        return jsonify({"error": "Missing required fields: 'query' and 'pdf_base64'"}), 400

    query = data["query"]
    pdf_b64 = data["pdf_base64"]

    try:
        text = extract_text_from_uploaded_pdf(pdf_b64)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    chunks = split_text(text)

    # ✅ Basic keyword filtering (optional)
    top_chunks = []
    query_keywords = set(re.findall(r'\w+', query.lower()))
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        if query_keywords & chunk_words:
            top_chunks.append(chunk)
        if len(top_chunks) >= 5:
            break
    if not top_chunks:
        top_chunks = chunks[:5]

    context = "\n".join(top_chunks)

    # ✅ Ask LLM (DeepSeek via OpenRouter)
    try:
        completion = openai.ChatCompletion.create(
            model="deepseek/deepseek-chat-v3-0324",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on insurance policy documents."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
            ]
        )
        return jsonify({"response": completion["choices"][0]["message"]["content"].strip()})
    except Exception as e:
        return jsonify({"error": f"LLM response error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
