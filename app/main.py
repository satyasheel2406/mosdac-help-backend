from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from .utils import load_knowledge_graph
from sentence_transformers import SentenceTransformer, util
import torch

# Load KG
KG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json"))
KG = load_knowledge_graph(KG_PATH)

# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 512  # Optional: Reduce memory
cleaned_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cleaned-txt"))

# Load cleaned-txt files
documents = []
file_sources = []

for filename in os.listdir(cleaned_folder):
    if filename.endswith(".txt"):
        path = os.path.join(cleaned_folder, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
            if text:
                documents.append(text)
                file_sources.append(filename)

# Compute embeddings
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# Setup FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "MOSDAC Help Bot Backend Running ✅"}

@app.get("/entities")
def get_entities():
    return KG["entities"]

@app.get("/relations")
def get_relations():
    return KG["relations"]

@app.get("/search")
def search(query: str):
    # === KG matching ===
    matches = [e for e in KG["entities"] if query.lower() in e["text"].lower()]
    matched_texts = {e["text"] for e in matches}
    relations = [r for r in KG["relations"] if r["source"] in matched_texts or r["target"] in matched_texts]

    # === Semantic answer from local cleaned-txt ===
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_idx = torch.argmax(scores).item()
    best_answer = documents[top_idx]
    best_score = float(scores[top_idx])

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "llm_response": best_answer if best_score > 0.4 else "No confident answer found.",
        "similarity_score": best_score
    }