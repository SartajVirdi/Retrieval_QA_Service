import os
import uuid
import fitz
import time
import cohere
from fastapi import FastAPI, Request
from pydantic import BaseModel
from chromadb.utils.embedding_functions import CohereEmbeddingFunction
import chromadb

# Load API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
assert COHERE_API_KEY, "COHERE_API_KEY not found in environment variables"

# Initialize components
embedder = CohereEmbeddingFunction(api_key=COHERE_API_KEY)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("policy_docs", embedding_function=embedder)
co = cohere.Client(COHERE_API_KEY)

# Ingest documents
def ingest():
    folder_path = "policies"
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        doc = fitz.open(pdf_path)
        docs, meta, ids = [], [], []
        for page in doc:
            text = page.get_text().strip()
            if len(text) < 50:
                continue
            docs.append(text)
            meta.append({"source": pdf_file})
            ids.append(str(uuid.uuid4()))
        for i in range(0, len(docs), 5):
            collection.add(
                documents=docs[i:i+5],
                metadatas=meta[i:i+5],
                ids=ids[i:i+5]
            )
            time.sleep(2)

# Initialize app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    ingest()

@app.post("/query")
def query_docs(request: QueryRequest):
    query_text = request.query
    results = collection.query(query_texts=[query_text], n_results=3)
    top_chunks = [doc for doc in results['documents'][0]]
    prompt = f"Answer this based on the documents: {query_text}\n\nContext:\n" + "\n".join(top_chunks)
    response = co.generate(prompt=prompt, model="command-r", max_tokens=300)
    return {"answer": response.generations[0].text.strip()}

# Render-compatible startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
