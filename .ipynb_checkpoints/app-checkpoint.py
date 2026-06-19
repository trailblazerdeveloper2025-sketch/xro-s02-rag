import streamlit as st
import os
import json
import time
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import storage

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "xro-lab")
STUDENT_ID = os.environ.get("STUDENT_ID", "student")
BATCH_ID = os.environ.get("BATCH_ID", "batch2")
GCS_BUCKET = "xro-lab-data"
REGION = "asia-south1"

st.set_page_config(page_title="XRO RAG Assistant", layout="wide")


@st.cache_resource
def load_resources():
    vertexai.init(project=PROJECT_ID, location=REGION)
    em = TextEmbeddingModel.from_pretrained("text-embedding-004")
    gm = GenerativeModel("gemini-1.5-flash")
    gcs = storage.Client(project=PROJECT_ID)
    blob = gcs.bucket(GCS_BUCKET).blob(f"{BATCH_ID}/{STUDENT_ID}/session3_vector_index.json")
    idx = json.loads(blob.download_as_text())["chunks"]
    return em, gm, idx


em, gm, vector_index = load_resources()


def cosine_sim(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rag_query(question, top_k=3):
    qv = em.get_embeddings([question])[0].values
    chunks = sorted(vector_index, key=lambda c: cosine_sim(qv, c["vector"]), reverse=True)[:top_k]
    ctx = "\n\n".join([f"[Chunk {i + 1}]\n{c['text']}" for i, c in enumerate(chunks)])
    prompt = f"Use ONLY this context:\n{ctx}\n\nQ: {question}\nA:"
    return gm.generate_content(prompt).text, chunks


st.title("XRO RAG Document Assistant")
st.caption(f"Student: {STUDENT_ID} | Project 1 -- Session 4")
st.divider()

question = st.text_input("Ask a question:", placeholder="What is RAG?")
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        answer, chunks = rag_query(question)
    st.success(answer)
    with st.expander("Source chunks"):
        for i, c in enumerate(chunks):
            st.markdown(f"**Chunk {i + 1}:** {c['text'][:200]}...")
