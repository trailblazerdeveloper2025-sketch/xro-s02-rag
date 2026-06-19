# {Project Name}

> One sentence: what this does and who it is for.

## Live Demo
https://YOUR-CLOUD-RUN-URL.run.app

## Architecture
User -> Cloud Run (Streamlit / FastAPI)
         |
     Vertex AI (text-embedding-004 + gemini-2.5-flash)
         |
     Cloud Storage (vector index)
         |
     Firestore (chat history)
         |
     BigQuery (analytics logs)

## What I Built
- Embedding pipeline that converts documents into a searchable vector index
- RAG query engine that retrieves relevant chunks and generates grounded answers
- Firestore memory for multi-turn conversations with persistent history
- Cloud Run deployment with auto-scaling and zero-downtime updates
- CI/CD pipeline -- every git push to main auto-deploys

## Tech Stack
| Layer    | Technology |
|----------|-----------|
| AI/ML    | Vertex AI, Gemini 2.5 Flash, text-embedding-004 |
| Backend  | FastAPI / Streamlit, Python 3.11 |
| Storage  | Cloud Storage (vectors), Firestore (memory), BigQuery (logs) |
| Deploy   | Docker, Cloud Run, Artifact Registry |
| CI/CD    | GitHub Actions, Cloud Build |
| Platform | Google Cloud Platform (asia-south1) |

## Built At
XRO Tech -- AI + Google Cloud Architecture Program -- Batch 2
xrotech.in -- Kokrajhar, Bodoland, Assam