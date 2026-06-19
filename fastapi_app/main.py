from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import time
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import storage, firestore

app = FastAPI(title="XRO RAG API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "xro-lab")
STUDENT_ID = os.environ.get("STUDENT_ID", "student")
BATCH_ID = os.environ.get("BATCH_ID", "batch2")
REGION = "asia-south1"


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    top_k: int = 3


class AskResponse(BaseModel):
    answer: str
    session_id: str
    tokens_used: int
    latency_ms: float


_resources = {}


def get_resources():
    if not _resources:
        vertexai.init(project=PROJECT_ID, location=REGION)
        _resources["em"] = TextEmbeddingModel.from_pretrained("text-embedding-004")
        _resources["gm"] = GenerativeModel("gemini-2.5-flash")
        gcs = storage.Client(project=PROJECT_ID)
        blob = gcs.bucket("xro-lab-data").blob(f"{BATCH_ID}/{STUDENT_ID}/session3_vector_index.json")
        _resources["idx"] = json.loads(blob.download_as_text())["chunks"]
    return _resources


def cosine_sim(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.get("/health")
def health():
    return {"status": "ok", "student": STUDENT_ID}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    r = get_resources()
    t0 = time.time()
    qv = r["em"].get_embeddings([req.question])[0].values
    chunks = sorted(r["idx"], key=lambda c: cosine_sim(qv, c["vector"]), reverse=True)[:req.top_k]
    ctx = "\n\n".join([f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks)])
    prompt = f"Use only this context:\n{ctx}\n\nQ: {req.question}\nA:"
    resp = r["gm"].generate_content(prompt)
    return AskResponse(
        answer=resp.text,
        session_id=req.session_id,
        tokens_used=resp.usage_metadata.total_token_count,
        latency_ms=round((time.time() - t0) * 1000, 1),
    )


@app.get("/conversations/{session_id}")
def get_conversation(session_id: str):
    db = firestore.Client(project=PROJECT_ID)
    docs = (
        db.collection("conversations")
        .document(f"{STUDENT_ID}_{session_id}")
        .collection("messages")
        .stream()
    )
    return {"messages": [{"role": d.get("role"), "content": d.get("content")} for d in docs]}
