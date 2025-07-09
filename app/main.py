from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
from .utils import load_knowledge_graph

# Load Hugging Face token
HF_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load KG
KG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG = load_knowledge_graph(KG_PATH)

# App setup
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Inference API (LLM Fallback)
def generate_llm_response(query: str) -> str:
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": f"[INST] You are a helpful assistant for the MOSDAC satellite portal. Answer this query: {query} [/INST]",
        "options": {"wait_for_model": True}
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        response_json = res.json()
        return response_json[0]["generated_text"]
    except Exception as e:
        return f"❌ Exception during LLM request: {str(e)}"

# Root endpoint
@app.get("/")
def root():
    return {"message": "MOSDAC Help Bot Backend Running"}

# Entity list
@app.get("/entities")
def get_entities():
    return KG["entities"]

# Relation list
@app.get("/relations")
def get_relations():
    return KG["relations"]

# Main search
@app.get("/search")
def search(query: str):
    matches = [e for e in KG["entities"] if query.lower() in e["text"].lower()]
    matched_texts = {e["text"] for e in matches}
    relations = [r for r in KG["relations"] if r["source"] in matched_texts or r["target"] in matched_texts]

    llm_response = generate_llm_response(query)

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "llm_response": llm_response
    }
