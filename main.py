import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from rag.db import get_collection
from rag.embeddings import embed_documents, embed_query, genai_client, GEMINI_MODEL
from rag.chunker import chunk_text, read_any_text
from rag.prompts import system_instruction, render_user_prompt

app = FastAPI(title="RAG Chatbot (Gemini + ChromaDB)")

# CORS (frontend can call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files correctly
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve index.html as the main UI"""
    return FileResponse(Path("static/index.html"))

# --- Session store ---
sessions: dict[str, list[dict]] = {}

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: int = 4

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict]

# --- Health check ---
@app.get("/health")
def health():
    return {"ok": True}

# --- File ingestion ---
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one file")

    collection = get_collection()
    added = 0

    for f in files:
        raw = await f.read()
        text = read_any_text(raw, filename=f.filename, content_type=f.content_type)
        chunks = chunk_text(text, chunk_size=900, overlap=200)
        if not chunks:
            continue

        embeddings = embed_documents(chunks)
        ids = [f"{f.filename}:{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": f.filename, "chunk": i, "size": len(chunks[i])}
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        added += len(chunks)
            # ✅ Instead of returning inside conversation, just return a clean notification
    return {"status": "ok", "message": f"✅ File(s) uploaded and indexed into {added} chunks."}


    return {"status": "ok", "chunks_added": added}

# --- Chat endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    collection = get_collection()
    count = len(collection.get().get("ids", []))
    if count == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Upload docs first.")

    q_vec = embed_query(req.message)
    results = collection.query(query_embeddings=[q_vec], n_results=max(1, req.top_k))

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[None] * len(docs)])[0]

    context_blocks = []
    for i, d in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist_info = f" | Distance: {dists[i]:.4f}" if dists and dists[i] is not None else ""
        tag = f"Source: {meta.get('source', 'unknown')} | Chunk: {meta.get('chunk', '?')}{dist_info}"
        context_blocks.append(f"[{tag}]\n{d}")
    context_text = "\n\n---\n\n".join(context_blocks)

    user_prompt = render_user_prompt(question=req.message, context=context_text)

    resp = genai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config={
            "system_instruction": system_instruction(),
            "max_output_tokens": 800,
            "temperature": 0.2,
        },
    )

    answer = None
    if hasattr(resp, "text"):
        answer = resp.text
    elif getattr(resp, "candidates", None):
        cands = resp.candidates
        answer = cands[0].get("content") if cands else None
    else:
        answer = str(resp)
    if not answer:
        answer = "(No text response from model)"

    sessions[session_id].append({"role": "user", "content": req.message})
    sessions[session_id].append({"role": "assistant", "content": answer})

    sources = []
    for i, meta in enumerate(metas):
        preview = docs[i][:160].replace("\n", " ") + ("..." if len(docs[i]) > 160 else "")
        sources.append({
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk", "?"),
            "preview": preview,
        })

    return ChatResponse(session_id=session_id, answer=answer, sources=sources)
