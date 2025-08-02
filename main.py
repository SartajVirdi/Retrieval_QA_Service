from flask import Flask, request, jsonify
import cohere
import fitz  # PyMuPDF
import numpy as np
import os
import re
import glob
import time
from sklearn.neighbors import NearestNeighbors

app = Flask(_name_)
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def load_all_pdfs(folder="policies"):
    combined_text = ""
    for filepath in glob.glob(os.path.join(folder, "*.pdf")):
        doc = fitz.open(filepath)
        for page in doc:
            combined_text += page.get_text() + "\n"
    return combined_text

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
    if not data or "query" not in data:
        return jsonify({"error": "Query field missing"}), 400

    query = data["query"]

    try:
        text = load_all_pdfs()
    except Exception as e:
        return jsonify({"error": f"Failed to load PDFs: {e}"}), 500

    chunks = split_text(text)

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

    try:
        nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        nn.fit(embeddings)
        query_embed = co.embed(texts=[query], model="embed-english-v3.0", input_type="search_query").embeddings[0]
        distances, indices = nn.kneighbors([query_embed])
        retrieved = [chunks[i] for i in indices[0]]
    except Exception as e:
        return jsonify({"error": f"Search error: {e}"}), 500

    context = "\n".join(retrieved)

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
