from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import os
import re
import glob
import google.generativeai as genai

app = Flask(__name__)

# Setup Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# Load all PDF content
def load_all_pdfs(folder="policies"):
    combined_text = ""
    for filepath in glob.glob(os.path.join(folder, "*.pdf")):
        doc = fitz.open(filepath)
        for page in doc:
            combined_text += page.get_text() + "\n"
    return combined_text

# Smarter chunking
def split_text(text, max_tokens=800, overlap=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, chunk = [], []
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
        full_text = load_all_pdfs()
        chunks = split_text(full_text)

        # Take top 5 chunks for now
        selected_context = "\n".join(chunks[:5])

        prompt = f"""
Based on the following policy content, answer the question below as clearly and accurately as possible.

Context:
{selected_context}

Question: {query}
Answer:"""

        response = model.generate_content(prompt)
        return jsonify({"response": response.text.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
