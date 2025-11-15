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


from typing import Dict

def weighted_jaccard(d1: Dict[int, float], d2: Dict[int, float]) -> float:
    common = d1.keys() & d2.keys()
    num = sum(min(d1[n], d2[n]) for n in common)
    denom = sum(d1.values()) + sum(d2.values()) - num
    return num / denom if denom > 0 else 0.0

def signed_weighted_jaccard(d1: Dict[int, float], d2: Dict[int, float]) -> float:
    # Séparer les positifs et les négatifs
    d1_pos = {k: v for k, v in d1.items() if v > 0}
    d1_neg = {k: -v for k, v in d1.items() if v < 0}  # valeurs rendues positives
    d2_pos = {k: v for k, v in d2.items() if v > 0}
    d2_neg = {k: -v for k, v in d2.items() if v < 0}

    # Calculer les Jaccard séparés
    j_pos = weighted_jaccard(d1_pos, d2_pos)
    j_neg = weighted_jaccard(d1_neg, d2_neg)

    # Pondérer par la "masse totale" des positifs/négatifs
    w_pos = sum(d1_pos.values()) + sum(d2_pos.values())
    w_neg = sum(d1_neg.values()) + sum(d2_neg.values())
    total = w_pos + w_neg

    if total == 0:
        return 0.0

    # Combinaison pondérée
    return (j_pos * w_pos + j_neg * w_neg) / total



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



