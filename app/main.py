from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
from .utils import load_knowledge_graph
import re
from difflib import SequenceMatcher

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

# Load cleaned-txt folder contents
CLEANED_TXT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned-txt")
doc_chunks = []
for fname in os.listdir(CLEANED_TXT_DIR):
    if fname.endswith(".txt"):
        with open(os.path.join(CLEANED_TXT_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = re.split(r"(?<=[.])\s+", text)
            doc_chunks.extend(chunks)

# Similarity
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
    return best_text.strip(), best_score

# OpenRouter AI Fallback
def generate_llm_response(query: str) -> str:
    api_key = os.getenv("API_KEY")
    if not api_key:
        return "❌ OpenRouter API key missing."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mosdac-help-frontend.vercel.app",  # ✅ Your frontend
        "X-Title": "MOSDAC Help Bot"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",  # ✅ lighter than GPT-4o
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for ISRO MOSDAC users."},
            {"role": "user", "content": query}
        ]
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        response_data = res.json()
        return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

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

    best_passage, score = find_best_passage(query)
    llm_response = generate_llm_response(query)

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "semantic_suggestions": [{"text": best_passage, "similarity": round(score, 2)}],
        "llm_response": llm_response
    }