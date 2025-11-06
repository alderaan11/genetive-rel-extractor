import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src2.models import Prep, Article, RelationInstance
from typing import Tuple


def build_encoders() -> Tuple[OneHotEncoder, OneHotEncoder]:
    prep_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    art_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    prep_enc.fit([[p.value] for p in Prep])
    art_enc.fit([[a.value] for a in Article])
    return prep_enc, art_enc


def encode_syntax(rel: RelationInstance, prep_enc: OneHotEncoder, art_enc: OneHotEncoder) -> np.ndarray:
    prep_vec = prep_enc.transform([[rel.prep.value]])[0]
    art_vec = (art_enc.transform([[rel.determinant.value]])[0]
               if rel.determinant else np.zeros(len(art_enc.categories_[0])))
    return np.concatenate([prep_vec, art_vec])


def make_feature_vector(
    rel: RelationInstance,
    simA: float, simB: float, score: float,
    prep_enc: OneHotEncoder, art_enc: OneHotEncoder
) -> np.ndarray:
    syntax = encode_syntax(rel, prep_enc, art_enc)
    return np.concatenate([[simA, simB, float(rel.is_det), score], syntax])