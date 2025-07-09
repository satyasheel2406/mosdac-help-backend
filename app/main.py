from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import requests
from dotenv import load_dotenv
from .utils import load_knowledge_graph
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()
HF_API_URL = os.getenv("HF_API_URL")
HF_API_KEY = os.getenv("HF_API_KEY")

# Load Knowledge Graph
KG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json"))
KG = load_knowledge_graph(KG_FILE)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a light model just for embedding
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
entity_texts = [ent["text"] for ent in KG["entities"]]
entity_embeddings = semantic_model.encode(entity_texts, convert_to_tensor=True)

# Query Hugging Face API (LLM only)
def generate_llm_response(query: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"You are an intelligent assistant for the MOSDAC satellite data portal. "
                  f"Answer the following query clearly, briefly, and helpfully:\n\n{query}"
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "").strip()
        return str(result)
    except Exception as e:
        return f"LLM Error: {str(e)}"

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
    from torch import tensor
    from torch.nn.functional import cosine_similarity

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
