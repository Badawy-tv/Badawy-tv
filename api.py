from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

app = FastAPI()

# Load embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare knowledge base
knowledge_dir = "knowledge"
documents = []
doc_texts = []

for filename in os.listdir(knowledge_dir):
    if filename.endswith(".txt"):
        path = os.path.join(knowledge_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            doc_texts.append(text)
            documents.append(text)

if documents:
    embeddings = model.encode(documents, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
else:
    index = None

@app.get("/")
def root():
    return {"message": "Badawy-tv AI backend running"}

@app.post("/query")
def query_ai(question: str):
    if not documents:
        return {"answer": "No knowledge available yet."}
    q_embedding = model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_embedding, k=1)
    answer = doc_texts[I[0][0]]
    return {"answer": answer}

@app.post("/task")
def run_task(task_type: str, input_text: str):
    # Placeholder for future tasks: summarization, QA, generation
    return {"task": task_type, "response": f"Task executed on: {input_text}"}
