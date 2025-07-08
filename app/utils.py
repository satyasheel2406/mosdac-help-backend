import json

def load_knowledge_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    return graph
