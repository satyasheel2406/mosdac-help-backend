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
    url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": f"Answer this helpdesk question for the MOSDAC user:\n{query}",
        "options": {"wait_for_model": True}
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        response_data = res.json()
        
        # Log full response for debugging
        print("🤖 LLM Response:", response_data)

        if isinstance(response_data, list) and "generated_text" in response_data[0]:
            return response_data[0]["generated_text"].strip()
        elif "error" in response_data:
            return f"⚠ Hugging Face error: {response_data['error']}"
        else:
            return "⚠ Unexpected response from AI model."
    except Exception as e:
        print("❌ LLM Fetch Error:", e)
        return "Sorry, I couldn't fetch a response now."

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
