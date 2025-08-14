# app.py
import io
import logging
import math
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BaseSettings, Field, HttpUrl, validator
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


# -----------------------------
# Settings & Configuration
# -----------------------------
class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = Field(..., env="OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = Field("anthropic/claude-3.5-sonnet", env="OPENROUTER_MODEL")
    OPENROUTER_BASE_URL: str = Field("https://openrouter.ai/api/v1/chat/completions", env="OPENROUTER_BASE_URL")

    # Retrieval/config knobs
    SENTENCE_MODEL_NAME: str = Field("all-MiniLM-L6-v2", env="SENTENCE_MODEL_NAME")
    MAX_PDF_MB: int = Field(20, env="MAX_PDF_MB")
    MAX_PDF_PAGES: int = Field(200, env="MAX_PDF_PAGES")
    CHUNK_CHAR_TARGET: int = Field(1200, env="CHUNK_CHAR_TARGET")  # aim for ~150-250 tokens
    CHUNK_CHAR_OVERLAP: int = Field(180, env="CHUNK_CHAR_OVERLAP")
    TOP_K: int = Field(3, env="TOP_K")  # number of chunks to send as context
    REQUEST_TIMEOUT_SECONDS: int = Field(60, env="REQUEST_TIMEOUT_SECONDS")
    ENABLE_CORS: bool = Field(True, env="ENABLE_CORS")

    class Config:
        case_sensitive = True


settings = Settings()

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("hackrx-app")
logger.info("App booting")

# Validate key explicitly for clear startup failure
if not settings.OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is required")


# -----------------------------
# HTTP Session with Retries
# -----------------------------
def build_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


SESSION = build_session()


# -----------------------------
# Sentence Transformer Singleton
# -----------------------------
class EmbeddingSingleton:
    _model: Optional[SentenceTransformer] = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._model is None:
            logger.info("Loading sentence-transformer model: %s", settings.SENTENCE_MODEL_NAME)
            cls._model = SentenceTransformer(settings.SENTENCE_MODEL_NAME)
            logger.info("Model loaded")
        return cls._model


# -----------------------------
# Schemas
# -----------------------------
class QARequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    top_k: Optional[int] = None  # override default TOP_K per-request
    max_pages: Optional[int] = None  # override default MAX_PDF_PAGES per-request

    @validator("questions")
    def non_empty_questions(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("questions must contain at least one question")
        return v


class ChunkEvidence(BaseModel):
    text: str
    score: float
    index: int


class SingleAnswer(BaseModel):
    answer: str
    evidence: List[ChunkEvidence]


class QAResponse(BaseModel):
    answers: Dict[str, SingleAnswer]


# -----------------------------
# Utilities
# -----------------------------
def _enforce_size_limit(resp: requests.Response, max_mb: int) -> bytes:
    size = int(resp.headers.get("Content-Length") or 0)
    if size and size > max_mb * 1024 * 1024:
        raise ValueError(f"PDF too large: {size / (1024 * 1024):.1f} MB exceeds limit of {max_mb} MB")

    # Stream with a hard cap, even if server omitted Content-Length
    cap = max_mb * 1024 * 1024
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=1 << 14):  # 16KB
        if chunk:
            if buf.tell() + len(chunk) > cap:
                raise ValueError(f"PDF download exceeded {max_mb} MB limit")
            buf.write(chunk)
    return buf.getvalue()


def fetch_pdf_bytes(url: str, timeout: int) -> bytes:
    logger.info("Downloading PDF from %s", url)
    headers = {"User-Agent": "hackrx-submission/1.0"}
    resp = SESSION.get(url, stream=True, timeout=timeout, headers=headers)
    if resp.status_code >= 400:
        raise ValueError(f"Failed to fetch PDF (status {resp.status_code})")
    return _enforce_size_limit(resp, settings.MAX_PDF_MB)


def extract_text_from_pdf_bytes(data: bytes, max_pages: int) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.page_count == 0:
            return ""
        if doc.page_count > max_pages:
            logger.warning("PDF has %d pages, truncating to %d", doc.page_count, max_pages)
        pages_to_read = min(doc.page_count, max_pages)
        texts = []
        for i in range(pages_to_read):
            page = doc.load_page(i)
            texts.append(page.get_text("text"))
        text = "\n".join(texts)
        logger.info("Extracted text from %d pages", pages_to_read)
        return text


def _merge_small_blocks(blocks: List[str], target_len: int, overlap: int) -> List[str]:
    merged: List[str] = []
    cursor = 0
    while cursor < len(blocks):
        cur = blocks[cursor]
        # keep extending until close to target_len
        j = cursor + 1
        while j < len(blocks) and len(cur) < target_len:
            nxt = blocks[j]
            # add a newline separator when merging paragraphs
            cur = (cur + "\n\n" + nxt).strip()
            j += 1
        merged.append(cur)

        if j >= len(blocks):
            break

        # compute char-based overlap move
        # back up by "overlap" within the freshly built chunk, approximated via step size
        step_chars = max(1, len(cur) - overlap)
        # estimate how many original blocks correspond to step_chars; fallback to 1
        acc = 0
        step_blocks = 0
        for k in range(cursor, j):
            acc += len(blocks[k]) + (2 if k > cursor else 0)
            step_blocks += 1
            if acc >= step_chars:
                break
        cursor = cursor + max(1, step_blocks)
    return merged


def chunk_text(text: str, target_chars: int, overlap_chars: int) -> List[str]:
    # Start with paragraph-level splits
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        return []

    # Merge into roughly target-sized chunks with overlap
    chunks = _merge_small_blocks(raw_paras, target_chars, overlap_chars)
    # Guardrails: drop ultra-short remnants and de-duplicate whitespace
    norm = [c.strip() for c in chunks if len(c.strip()) > 50]
    return norm


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def build_index(embeddings: np.ndarray) -> faiss.Index:
    # Use inner product over normalized vectors -> cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    return index


def top_k_search(
    model: SentenceTransformer,
    chunks: List[str],
    query: str,
    k: int,
) -> List[Tuple[int, float]]:
    # Embed chunks once per document
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
    chunk_embeddings = normalize(chunk_embeddings).astype("float32")

    # Build IP index
    index = build_index(chunk_embeddings)
    index.add(chunk_embeddings)

    # Embed query
    q = model.encode([query], convert_to_numpy=True)
    q = normalize(q).astype("float32")

    # Search
    k = max(1, min(k, len(chunks)))
    D, I = index.search(q, k)
    indices = I[0].tolist()
    scores = D[0].tolist()
    return list(zip(indices, scores))


def build_prompt(context_chunks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "You are an expert policy assistant. Use only the context to answer faithfully. "
        "If the answer is not in the context, say you cannot find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


def call_openrouter(prompt: str, timeout: int) -> str:
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "hackrx-submission",
    }
    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    resp = SESSION.post(settings.OPENROUTER_BASE_URL, json=payload, headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        raise ValueError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise ValueError(f"Unexpected OpenRouter response schema: {e}; body: {str(data)[:300]}")


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="HackRx Retrieval QA API", version="1.0.0")

if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/hackrx/run", response_model=QAResponse)
async def hackrx_query(payload: QARequest):
    try:
        max_pages = payload.max_pages or settings.MAX_PDF_PAGES
        top_k = payload.top_k or settings.TOP_K

        # 1) Fetch + parse PDF
        pdf_bytes = await run_in_threadpool(fetch_pdf_bytes, str(payload.documents), settings.REQUEST_TIMEOUT_SECONDS)
        text = await run_in_threadpool(extract_text_from_pdf_bytes, pdf_bytes, max_pages)
        if not text.strip():
            raise ValueError("No extractable text found in the PDF")

        # 2) Chunk document
        chunks = await run_in_threadpool(
            chunk_text, text, settings.CHUNK_CHAR_TARGET, settings.CHUNK_CHAR_OVERLAP
        )
        if not chunks:
            raise ValueError("Failed to create chunks from the PDF text")

        # 3) Get embedding model
        model = EmbeddingSingleton.get()

        # 4) Answer each question
        answers: Dict[str, SingleAnswer] = {}

        for q in payload.questions:
            results = await run_in_threadpool(top_k_search, model, chunks, q, top_k)

            # Collect top-k context and evidence
            top_indices = [idx for idx, _ in results]
            top_scores = [score for _, score in results]
            context_chunks = [chunks[i] for i in top_indices]

            prompt = build_prompt(context_chunks, q)
            llm_answer = await run_in_threadpool(call_openrouter, prompt, settings.REQUEST_TIMEOUT_SECONDS)

            evidence = [
                ChunkEvidence(text=chunks[i], score=float(s), index=int(i)) for i, s in zip(top_indices, top_scores)
            ]
            answers[q] = SingleAnswer(answer=llm_answer, evidence=evidence)

            logger.info("Answered question with top-%d context; best score=%.4f", len(results), max(top_scores) if top_scores else float("nan"))

        return QAResponse(answers=answers)

    except Exception as e:
        logger.error("Error during processing: %s", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
