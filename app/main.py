from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from .utils import load_knowledge_graph

# Load .env
load_dotenv()

HF_API_URL = os.getenv("HF_API_URL")
HF_API_KEY = os.getenv("HF_API_KEY")

# Load Knowledge Graph
KG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG_FILE = os.path.abspath(KG_FILE)
KG = load_knowledge_graph(KG_FILE)

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SentenceTransformer model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
entity_texts = [ent["text"] for ent in KG["entities"]]
entity_embeddings = semantic_model.encode(entity_texts, convert_to_tensor=True)

# Function to query Hugging Face Inference API
def generate_llm_response(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"You are an assistant for the MOSDAC satellite data portal. Answer this clearly and helpfully:\n\n{prompt}",
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        try:
            result = response.json()
            return result[0]["generated_text"].strip()
        except Exception:
            return "⚠ Error: Couldn't parse LLM response."
    else:
        return f"⚠ LLM API Error: {response.status_code}"

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
        fallback = [
            {"text": KG["entities"][i]["text"], "similarity": float(scores[i])}
            for i in scores.argsort(descending=True)[:3]
        ]

    llm_response = generate_llm_response(query)

    return {
        "query": query,
        "matches": matches,
        "related_relations": relations[:5],
        "semantic_suggestions": fallback,
        "llm_response": llm_response
    }
