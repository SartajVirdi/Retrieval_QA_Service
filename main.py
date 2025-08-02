# -- coding: utf-8 --

from flask import Flask, request, jsonify
import anthropic
import fitz  # PyMuPDF
import numpy as np
import os
import re
import time
import base64
from io import BytesIO

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ✅ Convert uploaded PDF (base64) to text
def extract_text_from_uploaded_pdf(b64_data):
    try:
        pdf_bytes = base64.b64decode(b64_data)
        pdf_file = BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"PDF processing error: {e}")

# ✅ Chunk text with overlap for context
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

    # ✅ Build context manually (top-k 5 chunks by length here, or implement similarity if needed)
    context = "\n\n".join(chunks[:5])  # You can implement similarity using external embedding API if needed

    # ✅ Ask Claude with prompt
    try:
        full_prompt = f"""You are a helpful assistant. Use the following insurance policy to answer the question.

Context:
{context}

Question: {query}
Answer:"""

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": full_prompt}]
        )

        return jsonify({"response": response.content[0].text.strip()})
    except Exception as e:
        return jsonify({"error": f"LLM response error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
