import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List
from src2.models import RelationInstance
from .similarity import triplet_similarity
from itertools import combinations
from tqdm import tqdm

def build_similarity_matrix(relations: List[RelationInstance], rel_id: int, cache_dir) -> np.ndarray:
    n = len(relations)
    matrix = np.zeros((n, n))

    for (i, t1), (j, t2) in tqdm(combinations(enumerate(relations), 2), total=n*(n-1)//2):
        _, _, score = triplet_similarity(t1, t2, rel_id, cache_dir)
        matrix[i, j] = matrix[j, i] = score
    np.fill_diagonal(matrix, 1.0)
    return matrix


def cluster_with_prototypes(
    triplets: List[RelationInstance],
    prototypes: List[RelationInstance],
    rel_id: int,
    cache_dir,
    k: int
):
    X = []
    for rel in triplets:
        vec = [(triplet_similarity(rel, proto, rel_id, cache_dir)[2]) for proto in prototypes]
        X.append(vec)
    X = np.array(X)
    kmeans = KMeans(n_clusters=k, random_state=11)
    labels = kmeans.fit_predict(X)
    return X, labels, kmeans


def plot_clusters(X, labels, prototypes, prototype_vecs, k: int):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    proto_pca = pca.transform(prototype_vecs)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7, s=50)
    plt.scatter(proto_pca[:, 0], proto_pca[:, 1], c='red', s=200, marker='X', label='Prototypes')
    plt.title(f"Clustering par similarité aux prototypes (k={k})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()