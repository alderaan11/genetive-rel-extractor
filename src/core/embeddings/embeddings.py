import typer
from pathlib import Path
from typing import List
from src.core.models.base_models import RelationInstance, RelProto
import json
from src.core.embeddings.features import build_encoders, get_jdm_relations, signed_weighted_jaccard, encode_syntax
from src.core.utils.loader import load_json_corpus
from tqdm import tqdm
import numpy as np
from itertools import zip_longest

app = typer.Typer()

def load_embeddings_from_dir(features_dir: Path):
    X_all, y_all = [], []
    rule_labels = None

    for feat_file in features_dir.glob("*_features.json"):
        with open(feat_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        X_all.extend(data["X"])
        y_all.extend(data["y"])

        # Toutes les features doivent avoir été générées avec la même signature
        if rule_labels is None:
            rule_labels = data["rule_ids"]

    X = np.array(X_all)
    y = np.array(y_all)
    # Reconstruction des noms des features
    # sims pour chaque type de règle
    feature_names = list(rule_labels)

    # On doit reconstruire la partie morpho-syntaxique
    prep_enc, art_enc = build_encoders()
    syntax_features = (
        prep_enc.get_feature_names_out().tolist() +
        art_enc.get_feature_names_out().tolist()
    )

    feature_names = feature_names + syntax_features

    return X, y, rule_labels, feature_names



@app.command("generate-embeddings")
def generate_embeddings(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
    rules_dir: Path = typer.Option(..., "--rules-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    jdm_rel_ids: List[int] = typer.Option([1, 6], "--jdm-rel", help="Liste d'IDs de relations JDM"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_enc, art_enc = build_encoders()

    all_rules: List[RelProto] = []
    for rule_file in sorted(rules_dir.glob("*_rules.json")):
        with open(rule_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_rules.extend(RelProto(**r) for r in data)

    typer.echo(f"{len(all_rules)} règles globales chargées")

    for json_file in corpus_dir.glob("*.json"):
        label = json_file.stem
        relations = load_json_corpus(json_file)
        typer.echo(f"Features pour {label}...")

        X, y = [], []

        for rel in tqdm(relations, desc=label):

            # Chargement des relations JDM pour chaque ID demandé
            relsA = [get_jdm_relations(rel.termA.name, rid, cache_dir) for rid in jdm_rel_ids]
            relsB = [get_jdm_relations(rel.termB.name, rid, cache_dir) for rid in jdm_rel_ids]

            # Similarités pour TOUTES les règles et TOUTES les relations JDM
            sims = []
            for rule in all_rules:
                sim_values = []
                for rel_a, rel_b in zip_longest(relsA, relsB, fillvalue=None):
                    simA = signed_weighted_jaccard(rel_a, rule.nodes_a) if rel_a else 0.0
                    simB = signed_weighted_jaccard(rel_b, rule.nodes_b) if rel_b else 0.0
                    sim_values.append((simA + simB) / 2)

                sims.append(np.mean(sim_values))


            # Ajout des infos syntaxiques  
            syntax_vec = encode_syntax(rel, prep_enc, art_enc)
            vec = np.concatenate([sims, syntax_vec])  #simalirité avec toutes les règles
            X.append(vec.tolist())
            y.append(label)

        
        feat_out = output_dir / f"{label}_features.json"
        with open(feat_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "X": X,
                    "y": y,
                    "rule_ids": [f"{r.gen_type}_{i}" for i, r in enumerate(all_rules)],
                    "jdm_rel_ids": jdm_rel_ids
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        typer.echo(f"Features sauvegardées : {feat_out}")



if __name__ == "__main__":
    app()
