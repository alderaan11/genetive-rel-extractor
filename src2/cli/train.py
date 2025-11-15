import typer
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from src2.models import RelationInstance, Article, Prep, TermInfo, RelProto
from src2.core.data import load_json_corpus
from src2.core.similarity import weighted_jaccard, get_jdm_relations, signed_weighted_jaccard
from src2.core.features import build_encoders, encode_syntax
from typing import List, Dict
import statistics

app = typer.Typer(help="Entraîne un arbre de décision avec explicabilité pour les relations génitives")



def create_rules_for_genitive_relation(
    rel_proto: List[RelProto],
    rel2: RelationInstance,
    traits_list: List[int],
    cache_dir: Path
) -> List[RelProto]:
    rel2_a, rel2_b = {}, {}

    # Récupération des relations JDM
    for trait in traits_list:
        rel2_a.update(get_jdm_relations(rel2.termA.name, trait, cache_dir))
        rel2_b.update(get_jdm_relations(rel2.termB.name, trait, cache_dir))

    # ✅ ici on itère directement sur la liste
    for rel in rel_proto:
        simA = signed_weighted_jaccard(rel.nodes_a, rel2_a)
        simB = signed_weighted_jaccard(rel.nodes_b, rel2_b)
        score_similarity = (simA + simB) / 2

        if score_similarity > 0.5:
            rel.nodes_a.update({
                key: statistics.mean([rel.nodes_a[key], rel2_a[key]]) if key in rel2_a else rel.nodes_a[key]
                for key in rel.nodes_a
            })
            rel.nodes_a.update({key: rel2_a[key] for key in rel2_a if key not in rel.nodes_a})

            rel.nodes_b.update({
                key: statistics.mean([rel.nodes_b[key], rel2_b[key]]) if key in rel2_b else rel.nodes_b[key]
                for key in rel.nodes_b
            })
            rel.nodes_b.update({key: rel2_b[key] for key in rel2_b if key not in rel.nodes_b})

            rel.fusion_number += 1
            return rel_proto

    # ✅ Si aucun prototype ne correspond
    rel_proto.append(
        RelProto(
            termA=rel2.termA.name,
            termB=rel2.termB.name,
            nodes_a=rel2_a,
            nodes_b=rel2_b,
            fusion_number=0
        )
    )
    return rel_proto


@app.command()
def debug():
    cache_dir = Path("./data2/cache")
    corpus_file = Path("./data2/json_corpus/r_depict.json")

    with open(corpus_file, "r", encoding="utf-8") as f:
        relations = load_json_corpus(corpus_file)
        label = corpus_file.stem

    # ✅ crée une vraie liste vide
    proto_rel: List[RelProto] = []

    traits_list = [1, 6]

    for rel in tqdm(relations, desc=f"Comparaison"):
        if not proto_rel:
            proto_rel.append(
                RelProto(
                    termA=rel.termA.name,
                    termB=rel.termB.name,
                    nodes_a=get_jdm_relations(rel.termA.name, traits_list[0], cache_dir),
                    nodes_b=get_jdm_relations(rel.termB.name, traits_list[0], cache_dir),
                    fusion_number=0
                )
            )
        else:
            proto_rel = create_rules_for_genitive_relation(proto_rel, rel, traits_list, cache_dir)

    print([rel.fusion_number for rel in proto_rel])



@app.command()
def train_genitif(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
    jdm_rel_id: int = typer.Option(6, "--jdm-rel"),
    output: Path = typer.Option("genitif_tree_model.pkl", "--output"),
    test_size: float = typer.Option(0.2, "--test"),
    max_depth: int = typer.Option(None, "--max-depth", help="Profondeur max de l'arbre"),
    viz: bool = typer.Option(True, "--viz", help="Afficher l'arbre"),
    rules: bool = typer.Option(True, "--rules", help="Afficher les règles textuelles"),
    importance: bool = typer.Option(True, "--importance", help="Afficher l'importance des features"),
):
    X_list, y_list = [], []
    prep_enc, art_enc = build_encoders()

    typer.echo("Chargement des données...")
    for json_file in corpus_dir.glob("*.json"):
        label = json_file.stem
        relations = load_json_corpus(json_file)
        if not relations:
            continue

        proto = relations[0]
        proto_a = get_jdm_relations(proto.termA.name, jdm_rel_id, cache_dir)
        proto_b = get_jdm_relations(proto.termB.name, jdm_rel_id, cache_dir)

        for rel in tqdm(relations, desc=f"Classe {label}"):
            rel_a = get_jdm_relations(rel.termA.name, jdm_rel_id, cache_dir)
            rel_b = get_jdm_relations(rel.termB.name, jdm_rel_id, cache_dir)
            simA = weighted_jaccard(rel_a, proto_a)
            simB = weighted_jaccard(rel_b, proto_b)
            vec = np.concatenate([[simA, simB, float(rel.is_det)], encode_syntax(rel, prep_enc, art_enc)])
            X_list.append(vec)
            y_list.append(label)

    if not X_list:
        typer.echo("Aucune donnée trouvée.")
        return

    X, y = np.array(X_list), np.array(y_list)
    feature_names = ['simA', 'simB', 'is_det'] + \
                    [f'prep_{p}' for p in prep_enc.categories_[0]] + \
                    [f'art_{a}' for a in art_enc.categories_[0]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
# clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    # --- Entraînement ---
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    typer.echo("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    # --- Sauvegarde ---
    model_data = {
        "model": clf,
        "prep_enc": prep_enc,
        "art_enc": art_enc,
        "jdm_rel_id": jdm_rel_id,
        "feature_names": feature_names
    }
    joblib.dump(model_data, output)
    typer.echo(f"Modèle sauvegardé : {output}")

    # ========================================
    # EXPLICABILITÉ
    # ========================================

    if importance:
        typer.echo("\nImportance des features :")
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in indices[:10]:  # Top 10
            typer.echo(f"  {feature_names[i]:<20} : {importances[i]:.4f}")

        # Graphique
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices[:15]], y=[feature_names[i] for i in indices[:15]])
        plt.title("Top 15 - Importance des features")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    if rules:
        typer.echo("\nRègles de l'arbre (texte) :")
        tree_rules = export_text(clf, feature_names=feature_names, max_depth=5)
        print(tree_rules)

    if viz:
        typer.echo("\nGénération de la visualisation de l'arbre...")
        dot_file = output.with_suffix(".dot")
        png_file = output.with_suffix(".png")

        export_graphviz(
            clf,
            out_file=str(dot_file),
            feature_names=feature_names,
            class_names=sorted(np.unique(y)),
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=4  # Limite affichage
        )

        try:
            import subprocess
            subprocess.run(["dot", "-Tpng", str(dot_file), "-o", str(png_file)], check=True)
            typer.echo(f"Arbre visualisé sauvegardé : {png_file}")
            # Optionnel : ouvrir l'image
            # import webbrowser; webbrowser.open(str(png_file))
        except Exception as e:
            typer.echo(f"Impossible de générer PNG (graphviz manquant ?) : {e}")
            typer.echo(f"Fichier DOT disponible : {dot_file}")

    # --- Optionnel : SHAP (très puissant) ---
    try:
        import shap
        typer.echo("\nCalcul des SHAP values...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test[:100])  # Limite pour performance

        if len(np.unique(y)) == 2:
            shap.summary_plot(shap_values[1], X_test[:100], feature_names=feature_names, show=False)
        else:
            for i, class_name in enumerate(clf.classes_):
                shap.summary_plot(shap_values[i], X_test[:100], feature_names=feature_names, show=False)
                plt.title(f"SHAP - Classe: {class_name}")
                plt.show()

        plt.tight_layout()
        plt.show()
    except ImportError:
        typer.echo("SHAP non disponible (installe avec `pip install shap`)")
    except Exception as e:
        typer.echo(f"Erreur SHAP : {e}")


if __name__ == "__main__":
    app()