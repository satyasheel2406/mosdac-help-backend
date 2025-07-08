import os
import re
import json

# ✅ Absolute paths for your project structure
CLEANED_DIR = r"D:\project-root\backend\data\cleaned-txt"
KEYWORD_FILE = r"D:\project-root\backend\data\keyword_list.txt"
KG_FILE = r"D:\project-root\backend\data\knowledge_graph.json"

graph = {
    "entities": [],
    "relations": []
}

# Load extracted keywords
with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
    KEYWORDS = [line.strip() for line in f.readlines()]

print(f"[✔] Loaded {len(KEYWORDS)} keywords from {KEYWORD_FILE}")

seen_entities = set()

def extract_entities(text):
    entities = []
    for kw in KEYWORDS:
        if re.search(rf'\b{re.escape(kw)}\b', text):
            entities.append(kw)
    return entities

# Process each text file
for filename in os.listdir(CLEANED_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(CLEANED_DIR, filename)
        print(f"[✔] Processing file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        entities = extract_entities(text)

        # Add unique entities
        for ent in entities:
            if ent not in seen_entities:
                seen_entities.add(ent)
                graph["entities"].append({"text": ent, "label": "KEYWORD"})

        # Basic consecutive relations
        for i in range(len(entities) - 1):
            relation = {
                "source": entities[i],
                "target": entities[i + 1],
                "type": "related_to"
            }
            graph["relations"].append(relation)

print(f"[✔] Extracted {len(graph['entities'])} unique entities")
print(f"[✔] Created {len(graph['relations'])} relations")

# Save KG
with open(KG_FILE, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2)

print(f"[✔] Knowledge Graph saved to {KG_FILE}")
