from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from .utils import load_knowledge_graph
from sentence_transformers import SentenceTransformer, util

# Paths
BASE_DIR = os.path.dirname(__file__)
KG_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "knowledge_graph.json"))
CLEANED_TXT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cleaned-txt"))

# Load KG
KG = load_knowledge_graph(KG_FILE)

# Load cleaned txt corpus
corpus = []
file_sources = []
for fname in os.listdir(CLEANED_TXT_DIR):
    path = os.path.join(CLEANED_TXT_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(line.strip())
                file_sources.append(fname)

# Load semantic model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = semantic_model.encode(corpus, convert_to_tensor=True)

# App
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
    return {"message": "MOSDAC Help Bot Backend Running"}

@app.get("/entities")
def get_entities():
    return KG["entities"]

@app.get("/relations")
def get_relations():
    return KG["relations"]

@app.get("/search")
def search(query: str):
    matches = [e for e in KG["entities"] if query.lower() in e["text"].lower()]
    matched_texts = {e["text"] for e in matches}
    relations = [r for r in KG["relations"] if r["source"] in matched_texts or r["target"] in matched_texts]

    # Fallback: semantic match from cleaned-txt if KG didn't help
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_idx = int(scores.argmax())
    fallback_answer = corpus[top_idx]
    fallback_file = file_sources[top_idx]

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "semantic_fallback": {
            "answer": fallback_answer,
            "source": fallback_file,
            "score": float(scores[top_idx])
        }
    }