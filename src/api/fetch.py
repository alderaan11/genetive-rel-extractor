import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import time
import typer

app = typer.Typer()

# API Base URL
API_BASE = "https://jdm-api.demo.lirmm.fr/v0"

def get_relation_type_id(relation_types: List[Dict], rel_name: str) -> int:
    """Find the ID for a given relation name (e.g., 'r_isa')."""
    for rel_type in relation_types:
        if rel_type.get('name') == rel_name:
            return rel_type['id']
    raise ValueError(f"Relation type '{rel_name}' not found.")

def fetch_node_name(node_id: int, cache: Dict[int, str]) -> str:
    """Fetch node name by ID using /v0/node_by_id/{node_id}, with caching."""
    if node_id in cache:
        return cache[node_id]
    
    url = f"{API_BASE}/node_by_id/{node_id}"
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

def fetch_node_relations_from(term: str, rel_type_id: int, node_cache: Dict[int, str], limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch outgoing relations of a specific type, return top 10 {name, weight} dicts by weight."""
    params = {
        'types_ids': [rel_type_id],
        'limit': limit,
        'min_weight': 1,  # Only relations with weight > 0
        'without_nodes': True  # Keep response small
    }
    url = f"{API_BASE}/relations/from/{term}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if 'relations' not in data:
            print(f"No relations found for '{term}' with type_id {rel_type_id}")
            return []
        
        results = []
        for rel in data['relations']:
            node2 = rel.get('node2')
            weight = rel.get('w', 0.0)
            if isinstance(node2, int):
                name = fetch_node_name(node2, node_cache)
                if name:
                    results.append({"name": name, "weight": weight})
            elif rel.get('node2_name'):
                results.append({"name": rel['node2_name'], "weight": weight})
        
        # Sort by weight (descending) and take top 10
        results = sorted(results, key=lambda x: x['weight'], reverse=True)[:10]
        return results
    except requests.RequestException as e:
        print(f"Error fetching relations for '{term}' (type_id {rel_type_id}): {e}")
        return []

def fetch_refinements(term: str) -> List[Dict[str, Any]]:
    """Fetch semantic refinements, return top 10 {name, weight} dicts by weight."""
    url = f"{API_BASE}/refinements/{term}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and 'nodes' in data:
            results = [{"name": node.get('name', ''), "weight": node.get('w', 0.0)} 
                       for node in data['nodes'] if node.get('name')]
        elif isinstance(data, list):
            results = [{"name": node.get('name', ''), "weight": node.get('w', 0.0)} 
                       for node in data if node.get('name')]
        else:
            print(f"Unexpected response format for refinements of '{term}': {data}")
            return []
        
        # Sort by weight (descending) and take top 10
        results = sorted(results, key=lambda x: x['weight'], reverse=True)[:10]
        return results
    except requests.RequestException as e:
        print(f"Error fetching refinements for '{term}': {e}")
        return []

@app.command()
def enrich_vocabulary(
    vocabulary_path: Path = typer.Argument(..., help="Path to input vocabulary.json"),
    enriched_path: Path = typer.Argument(..., help="Path to output enriched JSON"),
    delay: float = typer.Option(0.5, help="Delay (seconds) between API calls to avoid rate limiting")
):
    """
    Enrich the vocabulary.json by querying the JDM API for r_isa, r_raff_sem, and r_pos with weights.

    Args:
        vocabulary_path: Path to input vocabulary.json
        enriched_path: Path to output enriched JSON
        delay: Delay (seconds) between API calls
    """
    # Vérifier si le fichier d'entrée existe
    if not vocabulary_path.exists():
        print(f"❌ Input file not found: {vocabulary_path}")
        raise typer.Exit(code=1)

    # Créer le répertoire de sortie si nécessaire
    enriched_path.parent.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    with open(vocabulary_path, 'r', encoding='utf-8') as f:
        vocabulary_list: List[Dict[str, Any]] = json.load(f)

    # Fetch relation types once
    print("Fetching relation types...")
    try:
        resp = requests.get(f"{API_BASE}/relations_types", timeout=10)
        resp.raise_for_status()
        relation_types = resp.json()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch relation types: {e}")
        raise typer.Exit(code=1)

    # Get IDs for the required relations
    try:
        isa_id = get_relation_type_id(relation_types, 'r_isa')
        raff_sem_id = get_relation_type_id(relation_types, 'r_raff_sem')
        pos_id = get_relation_type_id(relation_types, 'r_pos')
        print(f"Relation IDs: r_isa={isa_id}, r_raff_sem={raff_sem_id}, r_pos={pos_id}")
    except ValueError as e:
        print(f"❌ Missing relation type: {e}")
        raise typer.Exit(code=1)

    # Node ID to name cache
    node_cache: Dict[int, str] = {}

    # Enrich each term
    total_terms = len(vocabulary_list)
    for i, entry in enumerate(vocabulary_list, 1):
        term = entry['name']
        print(f"Processing {i}/{total_terms}: '{term}'")

        # Fetch r_isa (outgoing hyperonymy relations)
        entry['r_isa'] = fetch_node_relations_from(term, isa_id, node_cache)

        # Fetch r_raff_sem (refinements)
        entry['r_raff_sem'] = fetch_refinements(term)

        # Fetch r_pos (part of speech relations)
        entry['r_pos'] = fetch_node_relations_from(term, pos_id, node_cache)

        print(entry)  # Debug print for each term
        # Rate limiting
        time.sleep(delay)

    # Save enriched vocabulary
    with open(enriched_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary_list, f, ensure_ascii=False, indent=2)
    
    print(f"✔️ Enriched vocabulary saved to {enriched_path}")
    print(f"  - Total terms: {total_terms}")
    print(f"  - Average r_isa: {sum(len(e['r_isa']) for e in vocabulary_list) / total_terms:.2f}")
    print(f"  - Average r_raff_sem: {sum(len(e['r_raff_sem']) for e in vocabulary_list) / total_terms:.2f}")
    print(f"  - Average r_pos: {sum(len(e['r_pos']) for e in vocabulary_list) / total_terms:.2f}")
    print(f"  - Node cache size: {len(node_cache)} names cached")

if __name__ == '__main__':
    app()