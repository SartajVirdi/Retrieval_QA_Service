from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import os
import re
import glob
import numpy as np
import faiss
import google.generativeai as genai

# Initialize Flask
app = Flask(__name__)

# Set Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load and combine all PDF text
def load_all_pdfs(folder="policies"):
    combined_text = ""
    for filepath in glob.glob(os.path.join(folder, "*.pdf")):
        doc = fitz.open(filepath)
        for page in doc:
            combined_text += page.get_text() + "\n"
    return combined_text

# Chunking logic: larger, smarter chunks with overlap
def split_text(text, max_words=300, overlap=75):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = []
    length = 0

    for sentence in sentences:
        words = sentence.split()
        if length + len(words) <= max_words:
            chunk.append(sentence)
            length += len(words)
        else:
            chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:] + [sentence]
            length = sum(len(s.split()) for s in chunk)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Embed using Gemini
def embed_texts(chunks):
    model = genai.GenerativeModel('embedding-001')
    response = model.embed_content(content=chunks, task_type="retrieval_document")
    return np.array(response["embedding"]).astype("float32")

# Flask webhook
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query field missing"}), 400

    query = data["query"]

    try:
        text = load_all_pdfs()
        chunks = split_text(text)
    except Exception as e:
        return jsonify({"error": f"PDF loading failed: {e}"}), 500

    # Embeddings & FAISS
    try:
        model = genai.GenerativeModel('embedding-001')
        chunk_embeds = [model.embed_content(c, task_type="retrieval_document")["embedding"] for c in chunks]
        chunk_embeds = np.array(chunk_embeds).astype("float32")

        index = faiss.IndexFlatL2(chunk_embeds.shape[1])
        index.add(chunk_embeds)

        # Embed query
        query_embed = model.embed_content(query, task_type="retrieval_query")["embedding"]
        D, I = index.search(np.array([query_embed], dtype="float32"), k=5)
        context = "\n".join([chunks[i] for i in I[0]])
    except Exception as e:
        return jsonify({"error": f"Embedding error: {e}"}), 500

    # Call Gemini Pro
    try:
        chat_model = genai.GenerativeModel("gemini-pro")
        chat = chat_model.start_chat()
        prompt = f"Based on the following content, answer the question:\n\nContent:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = chat.send_message(prompt)
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        return jsonify({"error": f"Gemini response error: {e}"}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
