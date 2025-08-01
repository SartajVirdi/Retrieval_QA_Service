import os
import fitz  # PyMuPDF
import uuid
import time
import cohere
from fastapi import FastAPI
from chromadb.utils.embedding_functions import CohereEmbeddingFunction
import chromadb

# Load Cohere API key from environment variable
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
assert COHERE_API_KEY, "COHERE_API_KEY not found in environment variables"

# Initialize Cohere embedding function
embedder = CohereEmbeddingFunction(api_key=COHERE_API_KEY)

# Initialize Chroma client
chroma_client = chromadb.Client()
chroma_client.heartbeat()

# Create or get collection
collection = chroma_client.get_or_create_collection("policy_docs", embedding_function=embedder)

# Ingest all PDF files from the 'policies/' folder
def ingest():
    folder_path = "policies"
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in 'policies/' folder.")
        return {"status": "No PDF files found."}

    batch_size = 5  # Lower batch size to avoid token overload
    delay_seconds = 2  # Time to wait between batches

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        pdf_path = os.path.join(folder_path, pdf_file)
        doc = fitz.open(pdf_path)

        docs, meta, ids = [], [], []
        for page in doc:
            text = page.get_text().strip()
            if len(text) < 50:
                continue  # skip very short pages
            docs.append(text)
            meta.append({"source": pdf_file})
            ids.append(str(uuid.uuid4()))

        # Add to collection in smaller batches
        for i in range(0, len(docs), batch_size):
            try:
                collection.add(
                    documents=docs[i:i+batch_size],
                    metadatas=meta[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
                print(f"Uploaded batch {i//batch_size + 1} of {pdf_file}")
                time.sleep(delay_seconds)
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1} of {pdf_file}: {e}")

    return {"status": "Ingestion completed."}


# FastAPI app setup
app = FastAPI()

@app.get("/")
def root():
    return {"message": "PDF Q&A API is running."}

@app.get("/ingest")
def run_ingestion():
    return ingest()


# Port binding block for Render
if __name__ == "__main__":
    import uvicorn
    ingest()
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
