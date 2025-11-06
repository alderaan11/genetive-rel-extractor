import typer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans

from src2.core.data import load_json_corpus
from src2.core.clustering import build_similarity_matrix

app = typer.Typer(help="Matrice de similarité + clustering")


@app.command()
def matrix_sim(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
    rel_id: int = typer.Option(6, "--rel-id"),
    k: int = typer.Option(5, "--k")
):
    relations = [rel for f in corpus_dir.glob("*.json") for rel in load_json_corpus(f)]
    sim_matrix = build_similarity_matrix(relations, rel_id, cache_dir)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(1 - sim_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis", square=True)
    plt.title("Matrice de similarité")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(labels)), [0]*len(labels), c=labels, cmap="tab10")
    plt.title("Clusters")
    plt.show()