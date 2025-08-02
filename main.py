from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

app = Flask(__name__)

# Utility: Load and chunk all PDFs
def extract_and_chunk_pdfs(folder_path, chunk_size=1000, overlap=200):
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, filename))
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            # Create overlapping chunks
            for i in range(0, len(full_text), chunk_size - overlap):
                chunk = full_text[i:i + chunk_size]
                if len(chunk.strip()) > 50:
                    chunks.append(chunk.strip())
    return chunks

# Preload chunks on app start
all_chunks = extract_and_chunk_pdfs("policies/")
chunk_embeddings = co.embed(texts=all_chunks, model="embed-multilingual-v3.0").embeddings

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        query_embedding = co.embed(texts=[query], model="embed-multilingual-v3.0").embeddings[0]
        sims = cosine_similarity([query_embedding], chunk_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:3]
        top_chunks = "\n\n".join([all_chunks[i] for i in top_indices])

        prompt = f"""Answer the following question based only on the policy text below.

Policy Content:
{top_chunks}

Question: {query}
Answer:"""

        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            temperature=0.3,
            max_tokens=300
        )
        return jsonify({"response": response.generations[0].text.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
