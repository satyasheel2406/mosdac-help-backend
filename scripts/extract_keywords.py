import os
import re
import spacy
from collections import Counter
import json

# Paths
TXT_DIR = "../data/cleaned-txt"
OUTPUT_PATH = "../data/keyword_list.txt"
KG_PATH = "../data/knowledge_graph.json"

# Load spaCy for entity typing
nlp = spacy.load("en_core_web_sm")

# Guess entity type using spaCy
def guess_entity_type(text):
    if "http" in text.lower() or text.lower().endswith(".gov.in"):
        return "Website"
    
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return "Organization"
        elif ent.label_ == "PRODUCT":
            return "Product"
        elif ent.label_ == "GPE":
            return "Location"
    return "Keyword"

# Extract keywords with frequency counts
def extract_keywords(text, min_freq=2):
    words = re.findall(r'\b[A-Za-z0-9\-/.]+\b', text)
    freq = Counter(words)
    return {word: count for word, count in freq.items() if count >= min_freq}

stopwords = {"the", "a", "an", "this", "that", "these", "those", "of", "in", "on", "at", "by", "for", "with", "and", "or", "to"}

# Extract relations based on improved patterns
def extract_relations(text):
    relations = []

    patterns = [
        (r'(\b[\w\-\/]+)\s+(?:operates|manages)\s+(\b[\w\-\/]+)', "operates"),
        (r'(\b[\w\-\/]+)\s+(?:provides|offers|delivers|contributes to|supports)\s+([\w\s\-\/]+?)(?:\s|\.|,)', "provides"),
        (r'(\b[\w\-\/]+)\s+(?:developed by|created by)\s+(\b[\w\-\/]+)', "developed_by"),
        (r'(\b[\w\-\/]+)\s+(?:located in|available in)\s+([\w\s\-\/]+?)(?:\s|\.|,)', "located_in"),
        (r'(\b[\w\-\/]+)\s+part of\s+([\w\s\-\/]+?)(?:\s|\.|,)', "part_of"),
        (r'(\b[\w\-\/]+)\s+(?:used for|capable of)\s+([\w\s\-\/]+?)(?:\s|\.|,)', "used_for"),
        (r'(\b[\w\-\/]+)\s+(?:handles|used by|enables|monitors)\s+([\w\s\-\/]+?)(?:\s|\.|,)', "enables"),
    ]

    for pattern, rel_type in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            source, target = match if isinstance(match, tuple) else (match, "")
            source = source.strip()
            target = target.strip().rstrip(".,")

            if source.lower() in stopwords or target.lower() in stopwords or not target:
                continue

            relations.append({"source": source, "target": target, "type": rel_type})

    return relations

# Dependency parsing-based relations
def extract_dependency_relations(text):
    relations = []
    doc = nlp(text)

    for sent in doc.sents:
        subj = None
        obj = None
        verb = None

        for token in sent:
            if "subj" in token.dep_:
                subj = token.text
            if "obj" in token.dep_:
                obj = token.text
            if token.pos_ == "VERB":
                verb = token.lemma_

        if subj and verb and obj and subj.lower() not in stopwords and obj.lower() not in stopwords:
            relations.append({"source": subj, "target": obj, "type": verb})

    return relations

def main():
    all_text = ""
    all_relations = []

    for filename in os.listdir(TXT_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(TXT_DIR, filename), "r", encoding="utf-8") as file:
                text = file.read()
                all_text += text + " "
                all_relations.extend(extract_relations(text))
                all_relations.extend(extract_dependency_relations(text))

    keywords_freq = extract_keywords(all_text)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        for word, count in keywords_freq.items():
            outfile.write(f"{word}\n")

    print(f"[✔] Extracted {len(keywords_freq)} keywords to {OUTPUT_PATH}")

    entities = []
    for keyword, count in keywords_freq.items():
        entity_type = guess_entity_type(keyword)
        entities.append({"text": keyword, "label": entity_type, "frequency": count})

    entity_texts = set(ent["text"] for ent in entities)

    filtered_relations = []
    for rel in all_relations:
        if rel["source"] in entity_texts and rel["target"] in entity_texts:
            filtered_relations.append(rel)

    print(f"[✔] Extracted {len(filtered_relations)} valid relations")

    kg = {"entities": entities, "relations": filtered_relations}

    with open(KG_PATH, "w", encoding="utf-8") as kgfile:
        json.dump(kg, kgfile, indent=2)

    print(f"[✔] Knowledge Graph saved to {KG_PATH}")

if __name__ == "__main__":
    main()
