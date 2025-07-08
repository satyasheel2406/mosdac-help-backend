from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from huggingface_hub import login
import torch
from .utils import load_knowledge_graph

# Load .env
load_dotenv()

# Hugging Face login
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)

# Load Knowledge Graph
KG_FILE = r"D:\project-root\backend\data\knowledge_graph.json"
KG = load_knowledge_graph(KG_FILE)

# FastAPI app
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load semantic search model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
entity_texts = [ent["text"] for ent in KG["entities"]]
entity_embeddings = semantic_model.encode(entity_texts, convert_to_tensor=True)

# Load Hugging Face LLM (Mistral)
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=0 if torch.cuda.is_available() else -1
)


# Generate LLM response
def generate_llm_response(query: str) -> str:
    prompt = (
        f"You are an intelligent assistant for the MOSDAC satellite data portal. "
        f"Answer the following query clearly, briefly, and helpfully:\n\n{query}"
    )
    result = llm(prompt, do_sample=True)[0]["generated_text"]
    return result.strip()

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
