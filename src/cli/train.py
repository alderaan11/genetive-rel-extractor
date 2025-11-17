from sklearn.ensemble import RandomForestClassifier
import typer
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from src.core.models.base_models import RelationInstance, RelProto
from src.core.utils.loader import load_json_corpus
from src.core.embeddings.features import get_jdm_relations, signed_weighted_jaccard
from src.core.embeddings.embeddings import load_embeddings_from_dir
from typing import List
import statistics
import subprocess
import optuna
import datetime
from src.core.utils.visualisation import plot_confusion_matrix, plot_top_features, plot_roc_curve
from src.core.utils.save import save_model, export_forest_tree


app = typer.Typer(help="Entraîne un arbre de décision avec explicabilité pour les relations génitives")


def rf_objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
    }

    clf = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    return f1_score(y_test, preds, average="macro")


@app.command("train-optuna")
def train_optuna(
    features_dir: Path = typer.Option(..., "--features-dir"),
    output_path: Path = typer.Option("./data2/models/rf_optuna_genitives.pkl", "--output"),
    n_trials: int = typer.Option(40, "--trials"),
    viz: bool = typer.Option(True, "--viz"),
):
    typer.echo(f"📂 Lecture des features depuis : {features_dir}")
    X, y, rule_labels = load_embeddings_from_dir(features_dir)

    typer.echo(f"\n📊 Dataset : {X.shape[0]} exemples | {X.shape[1]} dimensions | {len(set(y))} classes")

    # ---------------------- Optuna Optimization ----------------------
    typer.echo("\n🎯 Optimisation Optuna des hyperparamètres...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: rf_objective(trial, X, y), n_trials=n_trials)

    typer.echo("\n🏆 Meilleurs hyperparamètres trouvés :")
    best_params = study.best_params
    for k, v in best_params.items():
        typer.echo(f"  {k}: {v}")

    # ---------------------- Train Final Model ----------------------
    typer.echo("\n🌲 Entraînement du modèle final...")
    clf = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    typer.echo("\n📄 Rapport de classification :")
    print(classification_report(y_test, y_pred))

    # ---------------------- Confusion Matrix ----------------------
    if viz:
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        plt.figure(figsize=(11, 9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=clf.classes_, yticklabels=clf.classes_)
        plt.title("Matrice de confusion")
        plt.tight_layout()
        plt.show()

    # ---------------------- Feature Importances ----------------------
    if viz:
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        names = [rule_labels[i] if i < len(rule_labels) else f"feat_{i}" for i in idx]

        plt.figure(figsize=(10, 7))
        sns.barplot(x=importances[idx], y=names)
        plt.title("Top 20 Features Importantes")
        plt.tight_layout()
        plt.show()

    # ---------------------- Sauvegarde modèle ----------------------
    model_data = {
        "model": clf,
        "classes": clf.classes_.tolist(),
        "rule_labels": rule_labels,
        "hyperparams": best_params,
        "n_features": X.shape[1]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, output_path)

    typer.echo(f"\n💾 Modèle sauvegardé → {output_path}")

    # ---------------------- Export arbre ----------------------
    try:
        dot_path = output_path.with_suffix(".tree.dot")
        png_path = output_path.with_suffix(".tree.png")
        estimator = clf.estimators_[0]

        export_graphviz(
            estimator,
            out_file=str(dot_path),
            class_names=clf.classes_,
            filled=True, rounded=True,
            special_characters=True,
            max_depth=4
        )
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)])
        typer.echo(f"🌳 Arbre exporté : {png_path}")
    except Exception as e:
        typer.echo(f"⚠️ Export arbre impossible : {e}")

@app.command("train")
def train_model(
    embeddings_dir: Path = typer.Option(..., "--embeddings-dir"),
    test_size: float = typer.Option(0.2, "--test-size"),
    n_estimators: int = typer.Option(300, "--n-estimators"),
    max_depth: int = typer.Option(None, "--max-depth"),
    viz: bool = typer.Option(True, "--viz"),
):


    typer.echo(f"Chargement des features depuis : {embeddings_dir}")

    X, y, rule_labels, feature_names = load_embeddings_from_dir(embeddings_dir)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    typer.echo("\nEntraînement du modèle RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Classification 
    typer.echo("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    if viz:
        plot_confusion_matrix(y_test, y_pred, clf.classes_)
        plot_top_features(clf.feature_importances_, feature_names, top_k=33)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_test, y_score, clf.classes_)

    model_file = Path(f"rf_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.pkl")
    save_model(clf, model_file, rule_labels, X.shape[1])
    typer.echo(f"\nModèle sauvegardé → {model_file}")

    export_forest_tree(clf, model_file)
    typer.echo("\nEntraînement terminé.")





if __name__ == "__main__":
    app()