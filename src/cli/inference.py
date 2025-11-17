import typer
import joblib
import numpy as np
from pathlib import Path
from src.core.models.base_models import RelationInstance, TermInfo, Prep, Article, RelProto
from src.core.embeddings.features import signed_weighted_jaccard, get_jdm_relations, build_encoders, encode_syntax
import json
app = typer.Typer(help="🔮 Inférence interactive du type de relation génitive")

def load_all_rules(rules_dir: Path) -> list[RelProto]:
    all_rules = []
    for file in sorted(rules_dir.glob("*_rules.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = [RelProto(**r) for r in data]
        all_rules.extend(rules)

    return all_rules

def build_feature_vector_from_input(
    termA: str,
    prep: str,
    termB: str,
    rules_dir: Path,
    cache_dir: Path,
    jdm_rel_id: int = 6
) -> np.ndarray:
    """
    Reproduit EXACTEMENT la construction des vecteurs utilisée durant generate-features.
    """
    features_dir = Path("./data2/features")

    # 1) On charge rule_ids depuis n'importe quel fichier features
    any_feat = next(features_dir.glob("*_features.json"))
    with open(any_feat, "r", encoding="utf-8") as f:
        feat_data = json.load(f)

    rule_ids = feat_data["rule_ids"]  # même ordre que pour le training

    # 2) Encodeurs syntaxiques
    prep_enc, art_enc = build_encoders()

    # 3) Construire une RelationInstance minimale
    rel = RelationInstance(
        termA=TermInfo(name=termA),
        termB=TermInfo(name=termB),
        prep=Prep(prep.upper()) if prep.upper() in Prep.__members__ else Prep.DE,
        relation_type="?",
        is_det=False,
        determinant=None
    )

    # 4) Relations JDM
    rel_a = get_jdm_relations(termA, jdm_rel_id, cache_dir)
    rel_b = get_jdm_relations(termB, jdm_rel_id, cache_dir)

    sims = []

    # 5) Similarités avec toutes les règles individuelles
    for rid in rule_ids:
        gen_type, idx_str = rid.rsplit("_", 1)
        idx = int(idx_str)

        rule_file = rules_dir / f"{gen_type}_rules.json"
        with open(rule_file, "r", encoding="utf-8") as f:
            rules_json = json.load(f)

        rule = RelProto(**rules_json[idx])

        simA = signed_weighted_jaccard(rel_a, rule.nodes_a)
        simB = signed_weighted_jaccard(rel_b, rule.nodes_b)
        sims.append((simA + simB) / 2)

    # 6) Ajout syntaxe
    syntax_vec = encode_syntax(rel, prep_enc, art_enc)

    # 7) vecteur final shape (1, n_features)
    vec = np.concatenate([sims, syntax_vec]).reshape(1, -1)
    return vec

@app.command()
def infer(
    model_path: Path = typer.Option(..., "--model"),
    rules_dir: Path = typer.Option(..., "--rules-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
):
    """
    Lance une prédiction interactive :
      Ex:  > poème de poète
    """
    typer.echo("🔮 Chargement du modèle...")
    model_data = joblib.load(model_path)
    clf = model_data["model"]
    classes = model_data["classes"]

    typer.echo("Chargement des règles...")
    all_rules = load_all_rules(rules_dir)
    typer.echo(f"   → {len(all_rules)} règles chargées.")

    typer.echo(f"Modèle prêt ({len(classes)} classes apprises)")
    typer.echo("Format attendu : <termeA> <prep> <termeB>")
    typer.echo("   Exemple : poème de poète")
    typer.echo("   Tape 'exit' pour quitter\n")

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
            X_new = build_feature_vector_from_input(
                termA=termA,
                prep=prep,
                termB=termB,
                rules_dir= Path("./data2/rules"),
                cache_dir=cache_dir,
            )

            y_pred = clf.predict(X_new)[0]
            y_proba = clf.predict_proba(X_new)[0]

            top_idx = np.argsort(y_proba)[::-1][:3]
            typer.echo("\n🎯 Prédiction :")
            for i in top_idx:
                typer.echo(f"  {classes[i]:<25} → {y_proba[i]*100:5.2f}%")

            print()  # ligne vide

        except Exception as e:
            typer.echo(f"❌ Erreur durant l'inférence : {e}")



if __name__ == "__main__":
    app()
