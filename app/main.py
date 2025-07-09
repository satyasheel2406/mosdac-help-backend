from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
from .utils import load_knowledge_graph
from sentence_transformers import SentenceTransformer, util

# DO NOT run uvicorn from inside this file on Replit!

# Load KG
KG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG_FILE = os.path.abspath(KG_FILE)
KG = load_knowledge_graph(KG_FILE)

# Hugging Face Inference API token
HF_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small sentence transformer
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
entity_texts = [ent["text"] for ent in KG["entities"]]
entity_embeddings = semantic_model.encode(entity_texts, convert_to_tensor=True)

# Function to use Hugging Face Inference API
def generate_llm_response(query: str) -> str:
    api_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": f"You are a helpful assistant for the MOSDAC satellite portal.\n\nQuery: {query}",
        "options": {"wait_for_model": True}
    }
    response = requests.post(api_url, headers=headers, json=payload)
    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "Sorry, the LLM couldn't generate a response right now."

# Health check
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
def search_entity(query: str):
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, entity_embeddings)[0]

    top_indices = scores.argsort(descending=True)[:5]
    matches = [KG["entities"][i] for i in top_indices if scores[i] > 0.5]

    matched_texts = set(ent["text"] for ent in matches)
    relations = [
        rel for rel in KG["relations"]
        if rel["source"] in matched_texts or rel["target"] in matched_texts
    ]

    fallback = []
    if not matches:
        fallback = [{
            "text": KG["entities"][i]["text"],
            "similarity": float(scores[i])
        } for i in scores.argsort(descending=True)[:3]]

    llm_response = generate_llm_response(query)

    return {
        "query": query,
        "matches": matches,
        "related_relations": relations[:5],
        "semantic_suggestions": fallback,
        "llm_response": llm_response
    }
