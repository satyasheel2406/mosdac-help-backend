from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
from difflib import SequenceMatcher
from .utils import load_knowledge_graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load KG
KG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG = load_knowledge_graph(KG_PATH)

# Load cleaned-txt folder contents into memory
CLEANED_TXT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned-txt")
doc_chunks = []

for fname in os.listdir(CLEANED_TXT_DIR):
    if fname.endswith(".txt"):
        fpath = os.path.join(CLEANED_TXT_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
            # Split into sentences or chunks
            chunks = re.split(r"(?<=[.])\s+", text)
            doc_chunks.extend(chunks)

# Basic similarity function
def basic_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_passage(query):
    best_score = 0
    best_text = "Sorry, I couldn't find anything relevant."
    for chunk in doc_chunks:
        score = basic_similarity(query, chunk)
        if score > best_score:
            best_score = score
            best_text = chunk
    return best_text.strip()

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

    fallback = find_best_passage(query)

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "llm_response": fallback
    }