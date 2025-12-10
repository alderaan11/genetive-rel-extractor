import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.core.models.base_models import Prep, Article, RelationInstance
from typing import Tuple, Dict
from pathlib import Path
from src.core.models.base_models import ApiCall
import json 
from typing import Dict
from sklearn.preprocessing import OneHotEncoder
from src.core.utils.loader import get_jdm_relations
import math

def normalize_weight(w, method="tanh", alpha=0.01):
    if method == "tanh":  
        return math.tanh(alpha * w)
    elif method == "log":
        return math.copysign(math.log1p(abs(w)), w)
    return w


def build_encoders() -> Tuple[OneHotEncoder, OneHotEncoder]:
    def fit_encoder(values):
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
        enc.fit(np.array(values).reshape(-1, 1))
        return enc

    prep_enc = fit_encoder([p.value for p in Prep])
    art_enc = fit_encoder([a.value for a in Article])
    return prep_enc, art_enc


def encode_syntax(rel: RelationInstance, prep_enc: OneHotEncoder, art_enc: OneHotEncoder) -> np.ndarray:
    prep_vec = prep_enc.transform([[rel.prep.value]])[0]

    if rel.determinant:
        art_vec = art_enc.transform([[rel.determinant.value]])[0]
    else:
        art_vec = np.zeros(len(art_enc.categories_[0]), dtype=np.float32)

    return np.concatenate((prep_vec, art_vec))



# def make_feature_vector(
#     rel: RelationInstance,
#     simA: float, simB: float, score: float,
#     prep_enc: OneHotEncoder, art_enc: OneHotEncoder
# ) -> np.ndarray:
#     syntax = encode_syntax(rel, prep_enc, art_enc)
#     return np.concatenate([[simA, simB, float(rel.is_det), score], syntax])


def weighted_jaccard(d1: Dict[int, float], d2: Dict[int, float]) -> float:
    common = d1.keys() & d2.keys()
    num = sum(min(d1[n], d2[n]) for n in common)
    denom = sum(d1.values()) + sum(d2.values()) - num
    return num / denom if denom > 0 else 0.0

# def signed_weighted_jaccard(d1: Dict[int, float], d2: Dict[int, float]) -> float:
#     # Séparer les positifs et les négatifs
#     d1_pos = {k: v for k, v in d1.items() if v > 0}
#     d1_neg = {k: -v for k, v in d1.items() if v < 0}  
#     d2_pos = {k: v for k, v in d2.items() if v > 0}
#     d2_neg = {k: -v for k, v in d2.items() if v < 0}

    
#     j_pos = weighted_jaccard(d1_pos, d2_pos)
#     j_neg = weighted_jaccard(d1_neg, d2_neg)

    
#     w_pos = sum(d1_pos.values()) + sum(d2_pos.values())
#     w_neg = sum(d1_neg.values()) + sum(d2_neg.values())
#     total = w_pos + w_neg

#     if total == 0:
#         return 0.0

    
#     return (j_pos * w_pos + j_neg * w_neg) / total

def signed_weighted_jaccard(d1: Dict[int, float], d2: Dict[int, float],
                            normalize_weights=True, alpha=0.01) -> float:
    # optional weight normalization
    if normalize_weights:
        d1 = {k: math.tanh(alpha * v) for k, v in d1.items()}
        d2 = {k: math.tanh(alpha * v) for k, v in d2.items()}

    d1_pos = {k: v for k, v in d1.items() if v > 0}
    d1_neg = {k: -v for k, v in d1.items() if v < 0}
    d2_pos = {k: v for k, v in d2.items() if v > 0}
    d2_neg = {k: -v for k, v in d2.items() if v < 0}

    j_pos = weighted_jaccard(d1_pos, d2_pos)
    j_neg = weighted_jaccard(d1_neg, d2_neg)

    w_pos = sum(d1_pos.values()) + sum(d2_pos.values())
    w_neg = sum(d1_neg.values()) + sum(d2_neg.values())
    total = w_pos + w_neg

    if total == 0:
        return 0.0
    return (j_pos * w_pos - j_neg * w_neg) / total


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



