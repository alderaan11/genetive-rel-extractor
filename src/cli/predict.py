import typer
import joblib
import numpy as np
from pathlib import Path
import json
from itertools import zip_longest

from src.core.models.base_models import RelationInstance, TermInfo, Prep, RelProto
from src.core.embeddings.features import (
    signed_weighted_jaccard,
    get_jdm_relations,
    build_encoders,
    encode_syntax
)

app = typer.Typer(help="Inférence du type de relation génitive")


def load_all_rules(rules_dir: Path):
    all_rules = []
    for file in sorted(rules_dir.glob("*_rules.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_rules.extend(RelProto(**r) for r in data)
    return all_rules


def build_embedding_for_relation(
    termA: str,
    prep: str,
    termB: str,
    all_rules,
    jdm_rel_ids,
    cache_dir: Path
) -> np.ndarray:

    prep_enc, art_enc = build_encoders()

    rel = RelationInstance(
        termA=TermInfo(name=termA),
        termB=TermInfo(name=termB),
        prep=Prep(prep.upper()) if prep.upper() in Prep.__members__ else Prep.DE,
        relation_type="?",
        determinant=None,
        is_det=False
    )

    relsA = [get_jdm_relations(rel.termA.name, rid, cache_dir) for rid in jdm_rel_ids]
    relsB = [get_jdm_relations(rel.termB.name, rid, cache_dir) for rid in jdm_rel_ids]

    sims = []
    for rule in all_rules:
        sim_values = []
        for rel_a, rel_b in zip_longest(relsA, relsB, fillvalue=None):
            simA = signed_weighted_jaccard(rel_a, rule.nodes_a) if rel_a else 0.0
            simB = signed_weighted_jaccard(rel_b, rule.nodes_b) if rel_b else 0.0
            sim_values.append((simA + simB) / 2)
        sims.append(float(np.mean(sim_values)))

    syntax = encode_syntax(rel, prep_enc, art_enc)

    return np.concatenate([sims, syntax]).reshape(1, -1)


@app.command()
def infer(
    model_path: Path = typer.Option(..., "--model"),
    rules_dir: Path = typer.Option(..., "--rules-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
):
    """
    Inférence interactive :
      Exemple : peinture du peintre
    """
    typer.echo("Chargement du modèle...")
    model_data = joblib.load(model_path)
    clf = model_data["model"]
    classes = model_data["classes"]

    typer.echo("Chargement des règles...")
    all_rules = load_all_rules(rules_dir)
    typer.echo(f"   → {len(all_rules)} règles chargées.")

    if "jdm_rel_ids" in model_data:
        jdm_rel_ids = model_data["jdm_rel_ids"]
    else:
        typer.echo("Modèle ancien format → usage jdm_rel_ids = [6]")
        jdm_rel_ids = [1, 6]

    typer.echo(f"Modèle prêt ({len(classes)} classes)")
    typer.echo("Format attendu : <termeA> <prep> <termeB>")
    typer.echo("  Exemple : peinture du peintre")
    typer.echo("Tape 'exit' pour quitter.\n")

    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit"}:
            typer.echo("Fin de l'inférence.")
            break

        parts = text.split()
        if len(parts) < 3:
            typer.echo("Format incorrect. Exemple : table en bois")
            continue

        termA, prep, termB = parts[0], parts[1], parts[2]

        try:
            vec = build_embedding_for_relation(
                termA, prep, termB,
                all_rules=all_rules,
                jdm_rel_ids=jdm_rel_ids,
                cache_dir=cache_dir
            )

            y_pred = clf.predict(vec)[0]
            y_proba = clf.predict_proba(vec)[0]

            top_idx = np.argsort(y_proba)[::-1][:14]

            typer.echo("\nPrédiction :\n")
            for i in top_idx:
                typer.echo(f"  {classes[i]:<25} → {y_proba[i]*100:5.2f}%")
            print()

        except Exception as e:
            typer.echo(f"Erreur: {e}")


if __name__ == "__main__":
    app()
