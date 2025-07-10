from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os, json
from .utils import load_knowledge_graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Load Knowledge Graph ===
KG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_graph.json"))
KG = load_knowledge_graph(KG_PATH)

# === Load cleaned text files ===
CLEANED_TXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cleaned-txt"))
documents = []
doc_map = []

for fname in os.listdir(CLEANED_TXT_DIR):
    fpath = os.path.join(CLEANED_TXT_DIR, fname)
    if fname.endswith(".txt"):
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
            doc_map.append(fname)

# === Prepare TF-IDF ===
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(documents)

# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def search(query: str):
    # === KG Match ===
    matches = [e for e in KG["entities"] if query.lower() in e["text"].lower()]
    matched_texts = {e["text"] for e in matches}
    relations = [r for r in KG["relations"] if r["source"] in matched_texts or r["target"] in matched_texts]

    # === Lightweight NLP from cleaned-txt ===
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors)[0]
    top_indices = similarities.argsort()[::-1][:3]
    nlp_answers = [{"file": doc_map[i], "score": float(similarities[i]), "excerpt": documents[i][:300]} for i in top_indices]

    return {
        "query": query,
        "matches": matches[:5],
        "related_relations": relations[:5],
        "nlp_answers": nlp_answers
    }