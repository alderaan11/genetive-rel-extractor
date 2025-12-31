from typing import List
import typer
import joblib
import numpy as np
from pathlib import Path
from utils.logger import logger
from schemas.base_models import RelationInstance, TermInfo, Prep, Article, RelProto
from prepareembeddings.helper import signed_weighted_jaccard, get_jdm_relations, build_encoders, encode_syntax
from preparedataset.loader import parse_line
import json

app = typer.Typer(help="Inférence interactive du type de relation génitive")

def load_all_rules(rules_dir: Path) -> list[RelProto]:
    all_rules = []
    for file in sorted(rules_dir.glob("*_rules.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = [RelProto(**r) for r in data]
        all_rules.extend(rules)

    return all_rules

def build_feature_vector_from_input(
    relation_instance: RelationInstance,
    rules_dir: Path,
    embeddings_dir: Path,
    cache_dir: Path,
    jdm_rel_id: int = 6
) -> np.ndarray:
    """
    Reproduit EXACTEMENT la construction des vecteurs utilisée durant generate-features.
    """

    prep_enc, art_enc = build_encoders()
    rel_a = get_jdm_relations(relation_instance.termA, jdm_rel_id, cache_dir)
    rel_b = get_jdm_relations(relation_instance.termB, jdm_rel_id, cache_dir)
    
    sims = []

    # # 5) Similarités avec toutes les règles individuelles
    # for rid in rule_ids:
    #     gen_type, idx_str = rid.rsplit("_", 1)
    #     idx = int(idx_str)

    #     rule_file = rules_dir / f"{gen_type}_rules.json"
    #     with open(rule_file, "r", encoding="utf-8") as f:
    #         rules_json = json.load(f)

    #     rule = RelProto(**rules_json[idx])



    all_rules: List[RelProto] = []
    for rule_file in sorted(rules_dir.glob("*_rules.json")):
        with open(rule_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(data)
        all_rules.extend(RelProto(**r) for r in data)

    for rule in all_rules:
        simA = signed_weighted_jaccard(rel_a, rule.nodes_a)
        simB = signed_weighted_jaccard(rel_b, rule.nodes_b)
        sims.append((simA + simB) / 2)

    
    syntax_vec = encode_syntax(relation_instance, prep_enc, art_enc)

    # vecteur final shape (1, n_features)
    vec = np.concatenate([sims, syntax_vec]).reshape(1, -1)
    return vec

@app.command()
def infer(
    model_path: Path = typer.Option(..., "--model"),
    rules_dir: Path = typer.Option(..., "--rules-dir"),
    embeddings_dir: Path = typer.Option(..., "--embeddings-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
):
    """
    Lance une prédiction interactive :
      Ex:  > poème de poète
    """
    logger.info("Chargement du modèle...")
    model_data = joblib.load(model_path)
    clf = model_data["model"]
    classes = model_data["classes"]

    logger.info("Chargement des règles...")
    all_rules = load_all_rules(rules_dir)
    logger.info(f"   → {len(all_rules)} règles chargées.")

    logger.info(f"Modèle prêt ({len(classes)} classes apprises)")
    logger.info("Format attendu : <termeA> <prep> <termeB>")
    logger.info("   Exemple : poème de poète")
    logger.info("   Tape 'exit' pour quitter\n")

    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit"}:
            logger.info("Fin de l'inférence.")
            break

        parts = text.split()
        if len(parts) < 3:
            logger.info("Format incorrect. Exemple : table en bois")
            continue

        relation_instance = parse_line(text)

        try:
            X_new = build_feature_vector_from_input(
                relation_instance=relation_instance,
                rules_dir=rules_dir,
                embeddings_dir=embeddings_dir,
                cache_dir=cache_dir,
            )

            y_pred = clf.predict(X_new)[0]
            y_proba = clf.predict_proba(X_new)[0]

            top_idx = np.argsort(y_proba)[::-1][:3]
            logger.info("\n Prédiction :")
            for i in top_idx:
                logger.info(f"  {classes[i]:<25} -> {y_proba[i]*100:5.2f}%")


        except Exception as e:
            logger.info(f"Erreur durant l'inférence : {e}")



if __name__ == "__main__":
    app()
