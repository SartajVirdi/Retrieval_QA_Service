# 📄 PDF Q&A API with Cohere

A lightweight Flask API that answers natural language queries based on a PDF (`policy.pdf`). Uses Cohere embeddings + FAISS for retrieval and `command-r-plus` for generation.

---

## 🛠️ Tech Stack

- 🧠 Cohere (`embed-english-v3.0`, `command-r-plus`)
- 🗃️ FAISS for vector search
- 🧾 PyMuPDF to parse PDFs
- 🌐 Flask API (Render deployment)

---

## 🔧 Setup

1. Upload `policy.pdf` in your repo.
2. Deploy on **Render** with:
   - Python ≥ 3.10
   - `requirements.txt`
   - Environment variable: `COHERE_API_KEY`
3. On Colab, run:
   ```bash
   !curl -X POST https://<render-url>/webhook \
     -H "Content-Type: application/json" \
     -d '{"query":"example query"}'
