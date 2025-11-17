import typer
import json
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from typing import List
import statistics

from src.core.models.base_models import RelationInstance, RelProto
from src.core.utils.loader import load_json_corpus
from src.core.embeddings.features import signed_weighted_jaccard, get_jdm_relations
from src.core.embeddings.features import build_encoders, encode_syntax


app = typer.Typer(help="Pipeline complet : génération de règles, features, et apprentissage des génitifs")

def create_rules_for_genitive_relation(
    rel_proto: List[RelProto],
    rel2: RelationInstance,
    traits_list: List[int],
    cache_dir: Path,
    threshold: float = 0.3,
) -> List[RelProto]:
    """Fusionne un triplet dans un ensemble de règles (RelProto) selon la similarité."""
    rel2_a, rel2_b = {}, {}

    label = rel2.relation_type

    # Récupération des traits JDM (plusieurs relations possibles)
    for trait in traits_list:
        rel2_a.update(get_jdm_relations(rel2.termA.name, trait, cache_dir))
        rel2_b.update(get_jdm_relations(rel2.termB.name, trait, cache_dir))

    # Comparaison avec chaque règle existante
    for rel in rel_proto:
        simA = signed_weighted_jaccard(rel.nodes_a, rel2_a)
        simB = signed_weighted_jaccard(rel.nodes_b, rel2_b)
        score = (simA + simB) / 2

        if score > threshold:
            # Fusion pondérée des traits
            for d, new_d in [(rel.nodes_a, rel2_a), (rel.nodes_b, rel2_b)]:
                for key, val in new_d.items():
                    if key in d:
                        d[key] = statistics.mean([d[key], val])
                    else:
                        d[key] = val
            rel.fusion_number += 1
            return rel_proto

    # Si aucun prototype ne correspond, on en crée un nouveau
    rel_proto.append(
        RelProto(
            gen_type=label,
            termA=rel2.termA.name,
            termB=rel2.termB.name,
            nodes_a=rel2_a,
            nodes_b=rel2_b,
            fusion_number=0
                    )
    )
    return rel_proto


@app.command("generate-rules")
def generate_rules(
    corpus_dir: Path = typer.Option(..., "--corpus-dir", help="Dossier des corpus JSON"),
    cache_dir: Path = typer.Option(..., "--cache-dir", help="Cache JDM"),
    output_dir: Path = typer.Option("./data2/rules", "--output-dir", help="Où sauvegarder les règles"),
    traits: List[int] = typer.Option([1, 6], "--traits", help="IDs des relations JDM à utiliser"),
):
    """Génère un ensemble de règles (RelProto) pour chaque type génitif."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in corpus_dir.glob("*.json"):
        label = json_file.stem
        typer.echo(f"=== Génération des règles pour {label} ===")

        relations = load_json_corpus(json_file)
        if not relations:
            typer.echo(f"Aucun triplet dans {json_file}")
            continue

        proto_rel: List[RelProto] = []

        for rel in tqdm(relations, desc=f"{label}"):
            proto_rel = create_rules_for_genitive_relation(proto_rel, rel, traits, cache_dir)

        typer.echo(f"{len(proto_rel)} règles générées pour {label}")

        rules_out = output_dir / f"{label}_rules.json"
        with open(rules_out, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in proto_rel], f, indent=2, ensure_ascii=False)

        typer.echo(f"Sauvegardé : {rules_out}")



if __name__ == "__main__":
    app()
