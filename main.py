# -- coding: utf-8 --

from flask import Flask, request, jsonify
import cohere
import fitz  # PyMuPDF
import numpy as np
import os
import re
import time
import io
import base64

app = Flask(__name__)

# Load API key from environment variable
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ✅ Load PDF from base64 string
def load_pdf_from_base64(pdf_base64):
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {e}")

# ✅ Smart chunking with overlap
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
        return jsonify({"error": "Missing 'query' or 'pdf_base64' in request"}), 400

    query = data["query"]
    pdf_base64 = data["pdf_base64"]

    # ✅ Load text from PDF content
    try:
        text = load_pdf_from_base64(pdf_base64)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    chunks = split_text(text)

    # ✅ Embed all chunks in batches
    try:
        embeddings = []
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            estimated_tokens = int(sum(len(c.split()) for c in batch) * 1.3)
            print(f"⏳ Batch {i // batch_size + 1}, ~{estimated_tokens} tokens")

            embed_response = co.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            embeddings.extend(embed_response.embeddings)

            if estimated_tokens > 10000:
                time.sleep(5)
            elif estimated_tokens > 5000:
                time.sleep(3)
            else:
                time.sleep(1.5)
        embeddings = np.array(embeddings).astype("float32")
    except Exception as e:
        return jsonify({"error": f"Embedding error: {e}"}), 500

    # ✅ Embed query and find best matching chunk manually
    try:
        query_embed = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]

        # Cosine similarity
        def cosine_sim(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scores = [cosine_sim(query_embed, emb) for emb in embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        context = "\n".join([chunks[i] for i in top_indices])
    except Exception as e:
        return jsonify({"error": f"Search error: {e}"}), 500

    # ✅ Ask LLM
    try:
        response = co.chat(
            model="command-r-plus",
            message=query,
            documents=[{"text": context}]
        )
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        return jsonify({"error": f"LLM response error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
