import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests

# Embeddings + Vector store
import faiss
from sentence_transformers import SentenceTransformer

# Optional LLM providers
USE_OPENAI = False
USE_OLLAMA = False
try:
    from openai import OpenAI
    USE_OPENAI = True
except Exception:
    USE_OPENAI = False

load_dotenv()

# --- Variant knobs (read from env) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("OVERLAP", "150"))
PROMPT_STRICT = int(os.getenv("PROMPT_STRICT", "1"))  # 0=base, 1=stricter rules

OLLAMA_NUM_PREDICT = int(os.getenv("NUM_PREDICT", "220"))
OLLAMA_TOP_K = int(os.getenv("TOP_K", "30"))
OLLAMA_TOP_P = float(os.getenv("TOP_P", "0.9"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.05"))


# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("./chess_guide.txt")
INDEX_PATH = Path("./index.faiss")
CHUNKS_PATH = Path("./chunks.json")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP
    text = text.replace("\r\n", "\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = min(start + chunk_size, len(p))
                chunks.append(p[start:end])
                if end == len(p): break
                start = end - overlap
    return chunks


# -----------------------------
# Embeddings
# -----------------------------
def build_embedder():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    # MiniLM outputs 384-d vectors by default
    return model

def embed_texts(model, texts: List[str]) -> np.ndarray:
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")

# -----------------------------
# Indexing
# -----------------------------
def ingest_corpus():
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}. Put your chess guide there."
    print("Reading chess_guide.txt ...")
    text = DATA_PATH.read_text(encoding="utf-8")
    print("Chunking ...")
    chunks = chunk_text(text)

    # Save chunk metadata
    meta = [{"id": i, "source": str(DATA_PATH), "text": c} for i, c in enumerate(chunks)]
    CHUNKS_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build embeddings + FAISS
    print("Loading embedding model:", EMBED_MODEL_NAME)
    model = build_embedder()
    print("Embedding chunks ...")
    X = embed_texts(model, [m["text"] for m in meta])

    dim = X.shape[1]
    print(f"Building FAISS index (dim={dim}) ...")
    index = faiss.IndexFlatIP(dim)  # cosine if normalized
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Done. Saved {len(chunks)} chunks, index to {INDEX_PATH} and metadata to {CHUNKS_PATH}")

# -----------------------------
# Retrieval
# -----------------------------
def load_index_and_meta():
    assert INDEX_PATH.exists() and CHUNKS_PATH.exists(), "Run --ingest first."
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, meta

def retrieve(query: str, k: int = 6) -> List[Dict[str, Any]]:
    index, meta = load_index_and_meta()
    model = build_embedder()
    q = embed_texts(model, [query])
    D, I = index.search(q, k)
    results = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx == -1: continue
        m = meta[int(idx)]
        results.append({
            "rank": rank,
            "score": float(score),
            "id": int(m["id"]),
            "source": m["source"],
            "text": m["text"]
        })
    return results

# -----------------------------
# Generation
# -----------------------------
def build_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for d in docs:
        context_blocks.append(f"[{d['rank']}] (chunk {d['id']})\n{d['text']}")
    ctx = "\n\n".join(context_blocks)

    base_rules = [
        "Answer ONLY from the CONTEXT. If missing, say: “Not in the provided context.”",
        "Be concise; use short bullets when helpful.",
        "Every factual sentence must end with bracket citations like [1] or [2][5]."
    ]
    strict_rules = [
        "If a named technique/phrase appears in CONTEXT (e.g., “build a bridge”, “third rank”), copy it verbatim.",
        "Each sentence must end with at most two citations and only numbers from [1..k].",
        "If CONTEXT contains words like draw/win/winning/drawing about the position, include a single-line 'Verdict: <Draw|Win|Unclear>' with a citation."
    ]
    rules = base_rules + (strict_rules if PROMPT_STRICT else [])

    rules_text = "\n- ".join(rules)
    prompt = f"""You are a precise chess tutor.
QUESTION:
{question}

CONTEXT:
{ctx}

Requirements:
- {rules_text}
"""
    return prompt


def generate_with_openai(prompt: str) -> str:
    global USE_OPENAI
    if not OPENAI_API_KEY:
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You answer strictly from the provided context."},
                {"role": "user", "content": prompt}
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[openai-error] {e}"

def generate_with_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "top_k": OLLAMA_TOP_K,
                    "top_p": OLLAMA_TOP_P,
                    "repeat_penalty": OLLAMA_REPEAT_PENALTY
                }
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
    except Exception:
        return None


def answer_question(question: str, k: int = 6) -> Dict[str, Any]:
    docs = retrieve(question, k=k)
    top_score = docs[0]["score"] if docs else 0.0
    confidence = "high" if top_score >= 0.65 else "medium" if top_score >= 0.5 else "low"
    prompt = build_prompt(question, docs)

    # Prefer OpenAI, else Ollama, else fallback to extractive
    if OPENAI_API_KEY:
        out = generate_with_openai(prompt)
        if out and not out.startswith("[openai-error]"):
            used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", out) if m.isdigit()})
            return {"answer": out, "citations": used or [d["rank"] for d in docs], "context": docs, "confidence": confidence}

    # Ollama path
    out = generate_with_ollama(prompt)
    if out:
        used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", out) if m.isdigit()})
        return {"answer": out, "citations": used or [d["rank"] for d in docs], "context": docs, "confidence": confidence}

    # Fallback: simple extractive "answer" (top chunks)
    joined = "\n\n".join(f"[{d['rank']}] {d['text']}" for d in docs)
    fallback = (
        "No LLM configured. Here are the top relevant snippets:\n\n" + joined +
        "\n\n(Configure OPENAI_API_KEY or run an Ollama model to enable generation.)"
    )
    return {"answer": fallback, "citations": [d["rank"] for d in docs], "context": docs}

# -----------------------------
# API + CLI
# -----------------------------
class AskRequest(BaseModel):
    question: str
    k: int = 6

app = FastAPI(title="Chess RAG")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(req: AskRequest):
    return answer_question(req.question, k=req.k)

def cli():
    parser = argparse.ArgumentParser(description="RAG over chess_guide.txt")
    parser.add_argument("--ingest", action="store_true", help="Build index from chess_guide.txt")
    parser.add_argument("--ask", type=str, help="Ask a question")
    parser.add_argument("-k", type=int, default=6, help="Top-k chunks")
    parser.add_argument("--serve", action="store_true", help="Run API server")
    args = parser.parse_args()

    if args.ingest:
        ingest_corpus()

    if args.ask:
        res = answer_question(args.ask, k=args.k)
        print("\n=== ANSWER ===\n")
        print(res["answer"])
        print("\n=== CITATIONS (ranks) ===\n", res["citations"])
        print("\n=== CONTEXT HEADS ===")
        for d in res["context"]:
            print(f"[{d['rank']}] chunk={d['id']} score={d['score']:.3f}")

    if args.serve:
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    cli()
