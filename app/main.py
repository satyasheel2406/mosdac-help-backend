from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
from .utils import load_knowledge_graph

# Load Hugging Face token
HF_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load KG (assumes data/knowledge_graph.json exists)
KG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG = load_knowledge_graph(KG_PATH)

# App setup
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fallback LLM response using Hugging Face API
def generate_llm_response(query: str) -> str:
    url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": f"Answer this for MOSDAC user:\n{query}"}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        return res.json()[0]["generated_text"]
    except:
        return "Sorry, I couldn't fetch a response now."

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
    # VERY BASIC: Just find substring matches from entities
    matches = [e for e in KG["entities"] if query.lower() in e["text"].lower()]
    matched_texts = {e["text"] for e in matches}
    relations = [r for r in KG["relations"] if r["source"] in matched_texts or r["target"] in matched_texts]

    # LLM fallback always
    llm_response = generate_llm_response(query)

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "llm_response": llm_response
    }
