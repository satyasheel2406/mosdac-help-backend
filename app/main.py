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
    url = "https://api-inference.huggingface.co/models/gpt2"  # Safe public model
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": query}

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        print("🔁 HF Raw Response:", res.text)  # Log the full response body
        result = res.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif "error" in result:
            return f"⚠ HF Error: {result['error']}"
        else:
            return "❓ Unexpected response format."
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
