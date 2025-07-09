from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
from .utils import load_knowledge_graph

# Load Hugging Face token from environment (Render > Environment)
HF_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load Knowledge Graph
KG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG = load_knowledge_graph(KG_PATH)

# FastAPI app setup
app = FastAPI()

# Enable CORS for frontend access (important for Vercel connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM fallback function
def generate_llm_response(query: str) -> str:
    url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"Answer this for a MOSDAC user: {query}"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)

        if response.status_code != 200:
            return f"❌ HF API Error: {response.status_code} - {response.text}"

        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            return f"⚠ Unexpected HF response format: {data}"

    except Exception as e:
        return f"❌ Exception during LLM request: {e}"

# Root health check
@app.get("/")
def root():
    return {"message": "MOSDAC Help Bot Backend Running"}

# Endpoint to return all entities
@app.get("/entities")
def get_entities():
    return KG["entities"]

# Endpoint to return all relations
@app.get("/relations")
def get_relations():
    return KG["relations"]

# Search endpoint
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
