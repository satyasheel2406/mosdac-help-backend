from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import os, json, requests
from .utils import load_knowledge_graph

# Load Knowledge Graph
KG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json")
KG_FILE = os.path.abspath(KG_FILE)
KG = load_knowledge_graph(KG_FILE)

# Hugging Face API Token
HF_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load semantic model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
entity_texts = [ent["text"] for ent in KG["entities"]]
entity_embeddings = semantic_model.encode(entity_texts, convert_to_tensor=True)

# Basic NLP via Hugging Face API
def generate_llm_response(query: str) -> str:
    url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": f"You are an intelligent assistant for MOSDAC. Answer this:\n\n{query}",
        "options": {"wait_for_model": True}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        return response.json()[0]["generated_text"]
    except Exception:
        return "Sorry, I couldn’t fetch a response right now."

@app.get("/")
def root():
    return {"message": "MOSDAC Help Bot Backend Running"}

@app.get("/search")
def search(query: str):
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, entity_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:5]
    matches = [KG["entities"][i] for i in top_indices if scores[i] > 0.5]

    matched_texts = set(ent["text"] for ent in matches)
    relations = [
        rel for rel in KG["relations"]
        if rel["source"] in matched_texts or rel["target"] in matched_texts
    ]

    llm_response = generate_llm_response(query)
    return {
        "matches": matches,
        "relations": relations[:5],
        "llm_response": llm_response
    }
