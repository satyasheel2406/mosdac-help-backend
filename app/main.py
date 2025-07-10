from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
from .utils import load_knowledge_graph

# Load KG from data/knowledge_graph.json
KG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG = load_knowledge_graph(KG_PATH)

# Load OpenRouter API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Setup FastAPI app
app = FastAPI()

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ LLM response from Qwen2.5 via OpenRouter.ai
import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def generate_llm_response(query: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-frontend-site.vercel.app",  # ✅ Change this
            "X-Title": "MOSDAC Help Bot"
        }
        payload = {
            "model": "qwen2.5-1.5b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for the MOSDAC satellite data portal."},
                {"role": "user", "content": query}
            ]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Exception during LLM request: {e}"

# Root route
@app.get("/")
def root():
    return {"message": "MOSDAC Help Bot Backend Running with OpenRouter LLM"}

# Entities
@app.get("/entities")
def get_entities():
    return KG["entities"]

# Relations
@app.get("/relations")
def get_relations():
    return KG["relations"]

# Search endpoint
@app.get("/search")
def search(query: str):
    # Simple string matching on KG entities
    matches = [e for e in KG["entities"] if query.lower() in e["text"].lower()]
    matched_texts = {e["text"] for e in matches}
    relations = [r for r in KG["relations"] if r["source"] in matched_texts or r["target"] in matched_texts]

    # Call OpenRouter-powered Qwen2.5 model
    llm_response = generate_llm_response(query)

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "llm_response": llm_response
    }