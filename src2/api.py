import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import time
import typer

app = typer.Typer()

API_URL_BASE = "https://jdm-api.demo.lirmm.fr/v0"


def fetch_node_name(node_id: int, cache: Dict[int, str]) -> str:
    if node_id in cache:
        return cache[node_id]
    
    url = f"{API_URL_BASE}/node_by_id/{node_id}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        name = data.get('name', '')
        cache[node_id] = name
        return name
    except requests.RequestException as e:
        print(f"Error fetching node name for ID {node_id}: {e}")
        return ''
    
def fetch_node_relation_from(term: str, rel_type_id: int, node_cache:Dict[int, str]):
    params = {'types_ids': [rel_type_id]}

    url = f"{API_URL_BASE}/relations/from/{term}"

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(data)
        if 'relations' not in data:
            print(f"No relations found for '{term}' with type_id {rel_type_id}")
            return []
    
        results = []
        for rel in data['relations']:
            node2 = rel.get('node2')
            weight = rel.get('w')
            results.append({"name": node2, "weight": weight})
        
        results = sorted(results, key=lambda x: x["weight"], reverse=True)

    except requests.RequestException as e:
        print(f"Error fetching relations for '{term}' (type_id {rel_type_id}): {e}")
        return []
    

@app.command()
def enrich_vocabulary(
    vocabulary_path: Path = typer.Argument(..., help="Path to input vocabulary.json"),
    enriched_dir: Path = typer.Argument(..., help="Path to output enriched JSON"),
    type_id: int = typer.Argument(..., help="Id of the relation fetched"),
    delay: float = typer.Option(0.5, help="Delay (seconds) between API calls to avoid rate limiting")
):
    if not vocabulary_path.exists(): raise typer.Exit(code=1)

    enriched_dir.mkdir(parents=True, exist_ok=True)

    with open(vocabulary_path, "r", encoding="utf-8") as f:
        vocabulary_list = json.load(f)

    node_cache = {}
    total_terms = len(vocabulary_list)
    vocab_enriched = {}

    for i, term in enumerate(vocabulary_list, 1):
        print(f"Processing {i}/{total_terms}: {term}")
        relations = fetch_node_relation_from(term, type_id, node_cache)
        vocab_enriched[term] = relations
        print(vocab_enriched[term])
        time.sleep(delay)

    enriched_path = enriched_dir / f"vocab_{type_id}.json"
    with open(enriched_path, 'w', encoding="utf-8") as f:
        json.dump(vocab_enriched, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary enrichment finished. Saved to {enriched_path}")


if __name__=='__main__':
    app()