from gensim.models import FastText
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import pandas as pd
import typer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

app = typer.Typer()

def norm(corpus_path: Path) -> List[Dict[str, Any]]:
    with open(corpus_path) as f:
        corpus = json.load(f)
    data = []
    for row in corpus:
        data.append({
            "termA": row["termA"]["name"],
            "termB": row["termB"]["name"],
            "prep": row["prep"],
            "rel_type": row["rel_type"]
        })
    return data

def embedding(data: List[List[str]]) -> FastText:
    model = FastText(
        sentences=data,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,
        workers=4
    )
    return model

def phrase_to_vec(model: FastText, phrase: str) -> np.ndarray:
    tokens = phrase.split()
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

def visualize_embeddings3d(model: FastText):
    words = list(model.wv.index_to_key)
    word_vectors = np.array([model.wv[word] for word in words])
    tsne = TSNE(n_components=3, random_state=42, perplexity=15)
    word_vec_3d = tsne.fit_transform(word_vectors)

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(word_vec_3d[:,0], word_vec_3d[:,1], word_vec_3d[:,2], alpha=0.7)
    for i, word in enumerate(words):
        ax.text(word_vec_3d[i,0], word_vec_3d[i,1], word_vec_3d[i,2], word)
    ax.set_title("Visualisation des embeddings FastText (t-SNE 3D)")
    plt.show()

@app.command()
def main(corpus_path: Path, corpus_path2: Path):
    data = norm(corpus_path)
    data2 = norm(corpus_path2)
    fast_text_model = embedding(data+data2)
    
    # Visualisation 3D
    # visualize_embeddings3d(fast_text_model)

    examples = [f"{row['termA']} {row['prep']} {row['termB']}" for row in data]
    labels = [row['rel_type'] for row in data]
    examples += [f"{row['termA']} {row['prep']} {row['termB']}" for row in data2]
    labels += [row['rel_type'] for row in data2]
    
        # Encoder les phrases
    X = np.array([phrase_to_vec(fast_text_model, ex) for ex in examples])
    y = np.array(labels)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    app()
