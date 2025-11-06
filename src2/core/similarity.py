from typing import Dict, Tuple
from pathlib import Path
from src2.models import ApiCall
import json 

def get_jdm_relations(term: str, rel_id: int, cache_dir: Path) -> Dict[int, float]:
    cache_file = cache_dir / f"infos_by_name_{rel_id}.json"
    if not cache_file.exists():
        return {}
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    api_infos = ApiCall(**cache_data)
    nodes = api_infos.relation_nodes.get(term, [])
    return {n.node2: n.weight for n in nodes}


def weighted_jaccard(dict1: Dict[int, float], dict2: Dict[int, float]) -> float:
    if not dict1 or not dict2:
        return 0.0
    common = dict1.keys() & dict2.keys()
    num = sum(min(dict1[n], dict2[n]) for n in common)
    denom = sum(dict1.values()) + sum(dict2.values()) - num
    return num / denom if denom > 0 else 0.0


def triplet_similarity(
    t1, t2, rel_id: int, cache_dir: Path
) -> Tuple[float, float, float]:
    nodes_a1 = get_jdm_relations(t1.termA.name, rel_id, cache_dir)
    nodes_a2 = get_jdm_relations(t2.termA.name, rel_id, cache_dir)
    nodes_b1 = get_jdm_relations(t1.termB.name, rel_id, cache_dir)
    nodes_b2 = get_jdm_relations(t2.termB.name, rel_id, cache_dir)

    simA = weighted_jaccard(nodes_a1, nodes_a2)
    simB = weighted_jaccard(nodes_b1, nodes_b2)
    return simA, simB, (simA + simB) / 2

