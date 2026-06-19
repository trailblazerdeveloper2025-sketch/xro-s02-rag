import streamlit as st
import os
import json
import time
import uuid
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import storage, firestore

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "xro-lab")
STUDENT_ID = os.environ.get("STUDENT_ID", "student")
BATCH_ID = os.environ.get("BATCH_ID", "batch2")
REGION = "asia-south1"

st.set_page_config(page_title="XRO Project 2", layout="wide")


@st.cache_resource
def load_all():
    vertexai.init(project=PROJECT_ID, location=REGION)
    em = TextEmbeddingModel.from_pretrained("text-embedding-004")
    gm = GenerativeModel("gemini-2.5-flash")
    db = firestore.Client(project=PROJECT_ID)
    gcs = storage.Client(project=PROJECT_ID)
    blob = gcs.bucket("xro-lab-data").blob(f"{BATCH_ID}/{STUDENT_ID}/session3_vector_index.json")
    idx = json.loads(blob.download_as_text())["chunks"]
    return em, gm, db, idx


em, gm, db, vector_index = load_all()


def cosine_sim(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_history(sid):
    docs = (
        db.collection("conversations")
        .document(f"{STUDENT_ID}_{sid}")
        .collection("messages")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(5)
        .stream()
    )
    return list(reversed([{"role": d.get("role"), "content": d.get("content")} for d in docs]))


def save_msg(sid, role, content):
    db.collection("conversations").document(f"{STUDENT_ID}_{sid}").collection("messages").document().set({
        "role": role,
        "content": content,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })


def rag(question, sid):
    qv = em.get_embeddings([question])[0].values
    chunks = sorted(vector_index, key=lambda c: cosine_sim(qv, c["vector"]), reverse=True)[:3]
    ctx = "\n\n".join([f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks)])
    hist = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in get_history(sid)])
    prompt = f"Context:\n{ctx}\n\nHistory:\n{hist}\n\nQ: {question}\nA:"
    ans = gm.generate_content(prompt).text
    save_msg(sid, "user", question)
    save_msg(sid, "assistant", ans)
    return ans


st.title("XRO RAG Assistant -- Project 2")
st.caption(f"Student: {STUDENT_ID} | With Firestore memory")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
sid = st.session_state.session_id
st.caption(f"Session: {sid}")

history = get_history(sid)
for msg in history:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.chat_message("user").write(prompt)
    with st.spinner("Thinking..."):
        answer = rag(prompt, sid)
    st.chat_message("assistant").write(answer)
