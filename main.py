from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cohere
import fitz  # PyMuPDF
import faiss
import numpy as np
import os
import re
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load Cohere API key
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ===========================
# PDF LOADING & CHUNKING
# ===========================

def load_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def split_text(text, max_tokens=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_tokens:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def embed_chunks(chunks):
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return np.array(response.embeddings).astype("float32")

def preprocess_policy_pdf(pdf_path="policy.pdf"):
    print("[INFO] Processing PDF:", pdf_path)
    text = load_pdf(pdf_path)
    chunks = split_text(text)
    embeddings = embed_chunks(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save data
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save("embeddings.npy", embeddings)
    faiss.write_index(index, "faiss.index")
    print("[INFO] Processing complete.")

# ===========================
# PDF UPLOAD ENDPOINT
# ===========================

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    pdf_file = request.files['pdf']
    filename = secure_filename(pdf_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    pdf_file.save(filepath)

    try:
        preprocess_policy_pdf(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": f"{filename} uploaded and processed successfully."})

# ===========================
# QUERY ENDPOINT
# ===========================

def load_index():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load("embeddings.npy")
    index = faiss.read_index("faiss.index")
    return chunks, embeddings, index

def search(query, k=3):
    query_embed = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]
    D, I = index.search(np.array([query_embed], dtype="float32"), k)
    return [chunks[i] for i in I[0]]

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query field missing"}), 400

    query = data["query"]
    try:
        retrieved = search(query)
        context = "\n".join(retrieved)

        response = co.chat(
            model='command-r-plus',
            message=query,
            documents=[{"text": context}]
        )
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# INIT ON START
# ===========================

if os.path.exists("faiss.index"):
    chunks, embeddings, index = load_index()
else:
    chunks, embeddings, index = [], None, None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
