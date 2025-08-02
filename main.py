# -- coding: utf-8 --

from flask import Flask, request, jsonify
import cohere
import fitz  # PyMuPDF
import numpy as np
import os
import re
import time
import base64
from io import BytesIO

app = Flask(_name_)
co = cohere.Client(os.getenv("COHERE_API_KEY"))

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

    # Embed chunks
    try:
        embeddings = []
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embed_response = co.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            embeddings.extend(embed_response.embeddings)
            time.sleep(2)  # avoid rate limit
        embeddings = np.array(embeddings).astype("float32")
    except Exception as e:
        return jsonify({"error": f"Embedding error: {e}"}), 500

    # Embed query and retrieve top-k
    try:
        query_embed = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]

        similarities = np.dot(embeddings, query_embed)
        top_indices = similarities.argsort()[-5:][::-1]
        top_chunks = [chunks[i] for i in top_indices]
    except Exception as e:
        return jsonify({"error": f"Search error: {e}"}), 500

    context = "\n".join(top_chunks)

    try:
        response = co.chat(
            model='command-r-plus',
            message=query,
            documents=[{"text": context}]
        )
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        return jsonify({"error": f"LLM response error: {e}"}), 500

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)
