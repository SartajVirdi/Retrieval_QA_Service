from flask import Flask, request, jsonify
import base64
import fitz  # PyMuPDF
import requests
import os

app = Flask(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # or replace with your key directly

# ✅ Extract text from base64-encoded PDF
def extract_text_from_uploaded_pdf(b64_data):
    try:
        pdf_bytes = base64.b64decode(b64_data)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"PDF processing error: {e}")

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

    # Prepare Gemini REST API payload
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{query}\n\n{text[:20000]}"  # Trim text to avoid overload
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"response": reply.strip()})
    except Exception as e:
        return jsonify({"error": f"Gemini response error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
