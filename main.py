# -- coding: utf-8 --

from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import base64
from io import BytesIO
import re
import google.generativeai as genai
import os

app = Flask(__name__)

# Set your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# ✅ Extract text from base64 PDF
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

# ✅ Chunking
def split_text(text, max_tokens=500, overlap=100):
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
    if not data or "query" not in data or "pdf_base64" not in data:
        return jsonify({"error": "Missing 'query' or 'pdf_base64'"}), 400

    query = data["query"]
    pdf_b64 = data["pdf_base64"]

    try:
        text = extract_text_from_uploaded_pdf(pdf_b64)
        chunks = split_text(text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 🧠 Construct final prompt
    context = "\n\n".join(chunks[:10])  # Keep top 10 chunks to stay within limits
    prompt = f"""You are a helpful assistant answering questions from a policy document.
Document content:
{context}

Question: {query}
Answer:"""

    try:
        response = model.generate_content(prompt)
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        return jsonify({"error": f"Gemini response error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
