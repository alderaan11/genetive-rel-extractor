from pathlib import Path
import joblib
from pathlib import Path
import subprocess
from sklearn.tree import export_graphviz
import typer



def save_model(clf, output_path: Path, rule_labels, n_features: int):
    """
    Sauvegarde un modèle RandomForest avec ses métadonnées.
    """
    model_data = {
        "model": clf,
        "classes": clf.classes_.tolist(),
        "rule_labels": rule_labels,
        "n_features": n_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, output_path)

    return output_path



def export_forest_tree(clf, output_path: Path, max_depth: int = 4):
    """
    Exporte le premier arbre de la forêt en .dot et .png (Graphviz requis).
    """
    try:
        estimator = clf.estimators_[0]
        dot_path = output_path.with_suffix(".tree.dot")
        png_path = output_path.with_suffix(".tree.png")

        export_graphviz(
            estimator,
            out_file=str(dot_path),
            class_names=clf.classes_,
            filled=True, rounded=True,
            special_characters=True,
            max_depth=max_depth
        )
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)])
        typer.echo(f"Arbre exporté : {png_path}")

    except Exception as e:
        typer.echo(f"Erreur export arbre : {e}")
