import typer
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

from src2.core.data import load_json_corpus
from src2.core.similarity import triplet_similarity
from src2.core.features import build_encoders, make_feature_vector

app = typer.Typer(help="Moyenne incrémentale de similarité")


def incremental_mean_similarity(relations, rel_id, cache_dir):
    if len(relations) < 2:
        return [(0.0, 0.0, 0.0)] * len(relations)
    results = []
    ref = relations[0]
    meanA = meanB = mean_score = 0.0
    for i, rel in enumerate(relations[1:], start=2):
        simA, simB, score = triplet_similarity(rel, ref, rel_id, cache_dir)
        n = i - 1
        mean2 = lambda m, v: ((n-1) * m + v) / n
        meanA, meanB, mean_score = map(mean2, (meanA, meanB, mean_score), (simA, simB, score))
        results.append((meanA, meanB, mean_score))
        ref = rel
    return results


@app.command()
def mean_sim(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
    rel_id: int = typer.Option(6, "--rel-id")
):
    X, y = [], []
    prep_enc, art_enc = build_encoders()

    for file in tqdm(list(corpus_dir.glob("*.json"))):
        label = file.stem
        relations = load_json_corpus(file)
        if not relations:
            continue
        means = incremental_mean_similarity(relations, rel_id, cache_dir)
        for rel, (sA, sB, sc) in zip(relations[1:], means):
            X.append(make_feature_vector(rel, sA, sB, sc, prep_enc, art_enc))
            y.append(label)

    if not X:
        typer.echo("Aucune donnée.")
        raise typer.Exit(1)

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test)))