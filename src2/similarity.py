from pathlib import Path
import json
import typer
from typing import Dict
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src2.models import ApiCall, RelationInstance, TermInfo, Prep, Article
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
app = typer.Typer()


# ------------------ CHARGEMENT DES RELATIONS JDM ------------------

def get_relations_for_term_by_id(term: str, id_relation: int, api_call_cache_dir: Path) -> Dict[int, float]:
    """Charge les infos d’un terme (depuis le cache JDM) et renvoie {node2_id: weight}."""
    cache_file = api_call_cache_dir / f"infos_by_name_{id_relation}.json"

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache manquant pour le terme '{term}' ({cache_file})")

    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    api_infos = ApiCall(**cache_data)

    # Récupération des nœuds pour ce terme
    nodes_for_term = api_infos.relation_nodes.get(term, [])
    return {n.node2: n.weight for n in nodes_for_term}


def get_weighted_jaccard(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Calcule la similarité Jaccard pondérée entre deux ensembles de voisins."""
    if not a or not b:
        return 0.0
    common = a.keys() & b.keys()
    num = sum(min(a[n], b[n]) for n in common)
    denom = sum(a.values()) + sum(b.values()) - num
    return num / denom if denom else 0.0


def triplet_similarity(t1: RelationInstance, t2: RelationInstance, id_relation: int, api_call_cache_dir: Path) -> float:
    """Compare deux triplets (RelationInstance) à partir des graphes JDM."""
    nodes_term_a1 = get_relations_for_term_by_id(t1.termA.name, id_relation, api_call_cache_dir)
    nodes_term_a2 = get_relations_for_term_by_id(t2.termA.name, id_relation, api_call_cache_dir)
    nodes_term_b1 = get_relations_for_term_by_id(t1.termB.name, id_relation, api_call_cache_dir)
    nodes_term_b2 = get_relations_for_term_by_id(t2.termB.name, id_relation, api_call_cache_dir)

    simA = get_weighted_jaccard(nodes_term_a1, nodes_term_a2)
    simB = get_weighted_jaccard(nodes_term_b1, nodes_term_b2)
    return (simA + simB) / 2


# ------------------ ENCODAGE GRAMMATICAL ------------------

def build_encoders():
    """Construit les encodeurs OneHot pour Prep et Article."""
    prep_enc = OneHotEncoder(sparse_output=False)
    prep_enc.fit(np.array([[p.value] for p in Prep]))

    art_enc = OneHotEncoder(sparse_output=False)
    art_enc.fit(np.array([[a.value] for a in Article]))

    return prep_enc, art_enc


# ------------------ CONSTRUCTION DU VECTEUR DE FEATURES ------------------

def make_feature_vector(rel: RelationInstance, simA: float, simB: float, score: float,
                        prep_enc: OneHotEncoder, art_enc: OneHotEncoder):
    """Construit le vecteur de caractéristiques complet pour un triplet donné."""
    prep_vec = prep_enc.transform([[rel.prep.value]])[0]
    if rel.determinant:
        art_vec = art_enc.transform([[rel.determinant.value]])[0]
    else:
        art_vec = np.zeros(len(art_enc.categories_[0]))

    return np.concatenate([
        [simA, simB, rel.is_det, score],
        prep_vec,
        art_vec
    ])


# ------------------ COMMANDE DEBUG ------------------

@app.command()
def debug():
    """Teste la similarité entre deux génitifs et construit leur vecteur de caractéristiques."""
    t1 = RelationInstance(
        termA=TermInfo(name="peinture"),
        termB=TermInfo(name="peintre"),
        prep=Prep.DU,
        relation_type="r_auteur",
        is_det=False
    )
    t2 = RelationInstance(
        termA=TermInfo(name="poème"),
        termB=TermInfo(name="poète"),
        prep=Prep.DU,
        relation_type="r_auteur",
        is_det=False
    )

    api_call_cache_dir = Path("./data2/cache/")
    id_relation = 6  # ex: r_raff_sem, r_isa, etc.

    # Similarité sémantique
    score = triplet_similarity(t1, t2, id_relation, api_call_cache_dir)
    typer.echo(f"Similarité entre '{t1.termA.name} de {t1.termB.name}' et '{t2.termA.name} de {t2.termB.name}' : {score:.3f}")

    # Construction du vecteur de features
    simA, simB = 0.6, 0.8  # (valeurs simulées ici)
    prep_enc, art_enc = build_encoders()
    feature_vec = make_feature_vector(t1, simA, simB, score, prep_enc, art_enc)

    typer.echo(f"Vecteur de caractéristiques : {feature_vec}")
    typer.echo(f"Taille du vecteur : {len(feature_vec)}")


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    app()
