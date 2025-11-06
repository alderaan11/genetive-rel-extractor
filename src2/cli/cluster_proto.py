import typer
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src2.core.data import load_json_corpus
from src2.core.similarity import triplet_similarity
from src2.core.clustering import cluster_with_prototypes, plot_clusters

app = typer.Typer(help="Clustering par similarité aux prototypes")


@app.command()
def cluster_proto(
    corpus_dir: Path = typer.Option(..., "--corpus-dir"),
    cache_dir: Path = typer.Option(..., "--cache-dir"),
    rel_id: int = typer.Option(6, "--rel-id"),
    k: int = typer.Option(5, "--k"),
    plot: bool = typer.Option(True, "--plot")
):
    prototypes = {}
    all_triplets = []

    for file in corpus_dir.glob("*.json"):
        label = file.stem
        relations = load_json_corpus(file)
        if relations:
            prototypes[label] = relations[0]
            all_triplets.extend([(r, label) for r in relations])

    proto_list = list(prototypes.values())
    triplets = [t[0] for t in all_triplets]

    X, labels, kmeans = cluster_with_prototypes(triplets, proto_list, rel_id, cache_dir, k)

    if plot:
        proto_vecs = np.array([
            [triplet_similarity(p1, p2, rel_id, cache_dir)[2] for p2 in proto_list]
            for p1 in proto_list
        ])
        plot_clusters(X, labels, proto_list, proto_vecs, k)