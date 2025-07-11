from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
import requests
from dotenv import load_dotenv
from difflib import SequenceMatcher
from .utils import load_knowledge_graph

load_dotenv()

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
if os.path.exists(CLEANED_TXT_DIR):
    for fname in os.listdir(CLEANED_TXT_DIR):
        if fname.endswith(".txt"):
            fpath = os.path.join(CLEANED_TXT_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = re.split(r"(?<=[.])\s+", text)
                doc_chunks.extend(chunks)

# Simple string similarity
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

# OpenRouter API fallback
def query_openrouter_llm(prompt):
    api_key = os.getenv("API_KEY")
    if not api_key:
        return "❌ OpenRouter API key missing."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://mosdac-help-frontend.vercel.app",  # Update to your real URL
        "X-Title": "MOSDAC Help Bot"
    }

    payload = {
        "model": "qwen1.5-1.5b",  # You can change this to mistralai/mistral-7b-instruct or openai/gpt-4o
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for MOSDAC users."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
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

    # Cleaned-text fallback
    semantic = find_best_passage(query)

    # LLM fallback from OpenRouter
    llm = query_openrouter_llm(query)

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "semantic_suggestions": [{"text": semantic, "similarity": 1.0}] if semantic else [],
        "llm_response": llm
    }