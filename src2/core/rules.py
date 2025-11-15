import typer
import json
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import statistics

# === tes imports internes ===
from src2.models import RelationInstance, RelProto
from src2.core.data import load_json_corpus
from src2.core.similarity import weighted_jaccard, signed_weighted_jaccard, get_jdm_relations
from src2.core.features import build_encoders, encode_syntax


app = typer.Typer(help="Pipeline complet : génération de règles, features, et apprentissage des génitifs")


# =====================================================
# PHASE 1 — GÉNÉRATION DES RÈGLES PAR TYPE GÉNITIF
# =====================================================

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
            typer.echo(f"⚠️ Aucun triplet dans {json_file}")
            continue

        proto_rel: List[RelProto] = []

        for rel in tqdm(relations, desc=f"{label}"):
            proto_rel = create_rules_for_genitive_relation(proto_rel, rel, traits, cache_dir)

        typer.echo(f"{len(proto_rel)} règles générées pour {label}")

        rules_out = output_dir / f"{label}_rules.json"
        with open(rules_out, "w", encoding="utf-8") as f:
            json.dump([r.dict() for r in proto_rel], f, indent=2, ensure_ascii=False)

        typer.echo(f"✅ Sauvegardé : {rules_out}")


# =====================================================
# PHASE 2 — CALCUL DES SIMILARITÉS ET FEATURE VECTORS
# =====================================================

@app.command("generate-features")
def generate_features(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
    rules_dir: Path = typer.Option(..., "--rules-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
    output_dir: Path = typer.Option("./data2/features", "--output-dir"),
    jdm_rel_id: int = typer.Option(6, "--jdm-rel"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_enc, art_enc = build_encoders()

    # Charger toutes les règles en mémoire (toutes les classes confondues)
    all_rules: List[RelProto] = []
    for rule_file in rules_dir.glob("*_rules.json"):
        with open(rule_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        rules = [RelProto(**r) for r in data]
        all_rules.extend(rules)

    typer.echo(f"{len(all_rules)} règles globales chargées depuis {rules_dir}")

    # Calcul des features
    for json_file in corpus_dir.glob("*.json"):
        label = json_file.stem
        relations = load_json_corpus(json_file)
        typer.echo(f"Features pour {label}...")

        X, y = [], []

        for rel in tqdm(relations, desc=label):
            rel_a = get_jdm_relations(rel.termA.name, jdm_rel_id, cache_dir)
            rel_b = get_jdm_relations(rel.termB.name, jdm_rel_id, cache_dir)

            # Similarités avec TOUTES les règles individuelles
            sims = []
            for rule in all_rules:
                simA = signed_weighted_jaccard(rel_a, rule.nodes_a)
                simB = signed_weighted_jaccard(rel_b, rule.nodes_b)
                sims.append((simA + simB) / 2)

            # Ajout des infos syntaxiques (prep, article, is_det)
            syntax_vec = encode_syntax(rel, prep_enc, art_enc)
            vec = np.concatenate([sims, syntax_vec])
            X.append(vec.tolist())
            y.append(label)

        # Sauvegarde des features
        feat_out = output_dir / f"{label}_features.json"
        with open(feat_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "X": X,
                    "y": y,
                    "rule_ids": [f"{r.gen_type}_{i}" for i, r in enumerate(all_rules)]
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        typer.echo(f"✅ Features sauvegardées : {feat_out}")

# =====================================================
# PHASE 3 — APPRENTISSAGE (utilise les features JSON)
# =====================================================

@app.command("train")
def train_model(
    features_dir: Path = typer.Option(..., "--features-dir"),
    output: Path = typer.Option("genitif_tree.pkl", "--output"),
    test_size: float = typer.Option(0.2, "--test-size"),
    max_depth: int = typer.Option(None, "--max-depth"),
):
    """Entraîne un arbre de décision à partir des features sauvegardées."""
    X_all, y_all = [], []
    rule_labels = None

    for feat_file in features_dir.glob("*_features.json"):
        with open(feat_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        X_all.extend(data["X"])
        y_all.extend(data["y"])
        rule_labels = data["rule_labels"]

    X, y = np.array(X_all), np.array(y_all)
    feature_names = [f"sim_{r}" for r in rule_labels] + ["prep_DE", "prep_DU", "prep_DES", "prep_D", "is_det", "art_UN", "art_LE"]

    typer.echo(f"Dataset : {X.shape[0]} exemples, {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    typer.echo("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    # Importance des règles
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    typer.echo("\nTop 10 features :")
    for idx in sorted_idx[:10]:
        typer.echo(f"{feature_names[idx]} : {importances[idx]:.4f}")

    joblib.dump({"model": clf, "feature_names": feature_names}, output)
    typer.echo(f"✅ Modèle sauvegardé : {output}")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    app()
