from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
from .utils import load_knowledge_graph

# Load Hugging Face API token from environment variable
HF_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load Knowledge Graph JSON
KG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG = load_knowledge_graph(KG_PATH)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fallback LLM answer using Hugging Face
def generate_llm_response(query: str) -> str:
    url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": f"Answer this as a helpful assistant for the MOSDAC satellite portal:\n{query}",
        "options": {"wait_for_model": True}
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"]
    except Exception as e:
        return f"❌ Exception during LLM request: {str(e)}"

# Root route
@app.get("/")
def root():
    return {"message": "MOSDAC Help Bot Backend Running"}

# Entities route
@app.get("/entities")
def get_entities():
    return KG["entities"]

# Relations route
@app.get("/relations")
def get_relations():
    return KG["relations"]

# Search route
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