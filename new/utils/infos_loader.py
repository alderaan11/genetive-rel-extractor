from pathlib import Path
import json
from typing import List, Dict
from ..schemas.base_models import RelationInstance, ApiCall


def load_json_corpus(file_path: Path) -> List[RelationInstance]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [RelationInstance(**v) for v in data.get("data", {}).values()]

def get_jdm_relations(term: str, rel_id: int, cache_dir: Path) -> Dict[int, float]:
    cache_file = cache_dir / f"infos_by_name_{rel_id}.json"
    if not cache_file.exists():
        return {} 
    
    
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    api_infos = ApiCall(**cache_data)
    nodes = api_infos.relation_nodes.get(term, [])
    return {n.node2: n.weight for n in nodes}
