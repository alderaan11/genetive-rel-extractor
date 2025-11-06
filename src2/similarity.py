from pathlib import Path
import json
import typer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src2.models import ApiCall, RelationInstance, TermInfo, Prep, Article
from typing import List
from itertools import combinations
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import json
app = typer.Typer()


# ---------------------------------------------
# UTILITAIRES : JDM, SIMILARITÉS, ENCODAGE
# ---------------------------------------------

def get_relations_for_term_by_id(term: str, id_relation: int, api_call_cache_dir: Path) -> dict[int, float]:
    cache_file = api_call_cache_dir / f"infos_by_name_{id_relation}.json"
    if not cache_file.exists():
        return {}
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    api_infos = ApiCall(**cache_data)
    nodes_for_term = api_infos.relation_nodes.get(term, [])
    return {n.node2: n.weight for n in nodes_for_term}


def get_weighted_jaccard(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    common = a.keys() & b.keys()
    num = sum(min(a[n], b[n]) for n in common)
    denom = sum(a.values()) + sum(b.values()) - num
    return num / denom if denom else 0.0


def triplet_similarity(t1: RelationInstance, t2: RelationInstance, id_relation: int, api_call_cache_dir: Path):
    nodes_term_a1 = get_relations_for_term_by_id(t1.termA.name, id_relation, api_call_cache_dir)
    nodes_term_a2 = get_relations_for_term_by_id(t2.termA.name, id_relation, api_call_cache_dir)
    nodes_term_b1 = get_relations_for_term_by_id(t1.termB.name, id_relation, api_call_cache_dir)
    nodes_term_b2 = get_relations_for_term_by_id(t2.termB.name, id_relation, api_call_cache_dir)
    simA = get_weighted_jaccard(nodes_term_a1, nodes_term_a2)
    simB = get_weighted_jaccard(nodes_term_b1, nodes_term_b2)
    return simA, simB, (simA + simB) / 2

@app.command("train-genitif")
def train_genitif_command(
    corpus_dir: Path = typer.Option(..., "--corpus-dir", help="Dossier JSON d'entraînement"),
    api_call_cache_dir: Path = typer.Option(..., "--cache-dir", help="Cache JDM"),
    jdm_relation_id: int = typer.Option(6, "--jdm-rel", help="ID relation JDM (ex: r_isa = 6)"),
    model_output: Path = typer.Option("genitif_model.pkl", "--output", help="Fichier modèle"),
    test_size: float = typer.Option(0.2, "--test", help="Proportion test")
):
    """Entraîne un modèle statistique pour prédire les relations génitives."""
    
    typer.echo("Chargement des données...")
    X_list = []
    y_list = []

    prep_enc, art_enc = build_encoders()  # one-hot pour prep et determinant

    # --- 1. Pour chaque classe génitive ---
    for json_file in corpus_dir.glob("*.json"):
        label = json_file.stem
        with open(json_file) as f:
            data = json.load(f)
        relations = [RelationInstance(**v) for v in data.get("data", {}).values()]
        if not relations:
            continue

        # Prototype = 1er triplet
        proto = relations[0]

        # --- 2. Pour chaque triplet de la classe ---
        for rel in tqdm(relations):
            # JDM
            simA = get_weighted_jaccard(
                get_relations_for_term_by_id(rel.termA.name, jdm_relation_id, api_call_cache_dir),
                get_relations_for_term_by_id(proto.termA.name, jdm_relation_id, api_call_cache_dir)
            )
            simB = get_weighted_jaccard(
                get_relations_for_term_by_id(rel.termB.name, jdm_relation_id, api_call_cache_dir),
                get_relations_for_term_by_id(proto.termB.name, jdm_relation_id, api_call_cache_dir)
            )

            # Syntaxe
            prep_vec = prep_enc.transform([[rel.prep.value]])[0]
            art_vec = art_enc.transform([[rel.determinant.value]])[0] if rel.determinant else np.zeros(len(art_enc.categories_[0]))

            # Vecteur final
            feature_vec = np.concatenate([[simA, simB, float(rel.is_det)], prep_vec, art_vec])
            X_list.append(feature_vec)
            y_list.append(label)

    if not X_list:
        typer.echo("Aucune donnée trouvée.")
        return

    X = np.array(X_list)
    y = np.array(y_list)
    typer.echo(f"Dataset : {X.shape[0]} exemples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # --- 3. Train / Test ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # --- 4. Modèle ---
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --- 5. Évaluation ---
    typer.echo("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    # --- 6. Sauvegarde ---
    import joblib
    joblib.dump({
        "model": clf,
        "prep_enc": prep_enc,
        "art_enc": art_enc,
        "jdm_relation_id": jdm_relation_id,
        "feature_names": ['simA', 'simB', 'is_det'] + 
                        [f'prep_{p}' for p in prep_enc.categories_[0]] +
                        [f'art_{a}' for a in art_enc.categories_[0]]
    }, model_output)
    typer.echo(f"\nModèle sauvegardé : {model_output}")


def build_encoders():
    prep_enc = OneHotEncoder(sparse_output=False)
    prep_enc.fit(np.array([[p.value] for p in Prep]))
    art_enc = OneHotEncoder(sparse_output=False)
    art_enc.fit(np.array([[a.value] for a in Article]))
    return prep_enc, art_enc


def make_feature_vector(rel: RelationInstance, simA: float, simB: float, score: float,
                        prep_enc: OneHotEncoder, art_enc: OneHotEncoder) -> np.ndarray:
    prep_vec = prep_enc.transform([[rel.prep.value]])[0]
    if rel.determinant:
        art_vec = art_enc.transform([[rel.determinant.value]])[0]
    else:
        art_vec = np.zeros(len(art_enc.categories_[0]))
    return np.concatenate([[simA, simB, rel.is_det, score], prep_vec, art_vec])


# ---------------------------------------------
# MOYENNE DES SIMILARITÉS INTRA-TYPE
# ---------------------------------------------

def mean_intra_similarity(triplet: RelationInstance, others: List[RelationInstance],
                          id_relation: int, api_call_cache_dir: Path):
    simsA, simsB, scores = [], [], []
    for other in others:
        if other == triplet:
            continue
        simA, simB, score = triplet_similarity(triplet, other, id_relation, api_call_cache_dir)
        simsA.append(simA)
        simsB.append(simB)
        scores.append(score)
    if not simsA:
        return 0.0, 0.0, 0.0
    return np.mean(simsA), np.mean(simsB), np.mean(scores)


@app.command("mean-sim")
def mean_sim_command(
    corpus_dir: Path = typer.Option(..., help="Dossier contenant les fichiers JSON de corpus"),
    api_call_cache_dir: Path = typer.Option(..., help="Dossier cache des appels JDM"),
    id_relation: int = typer.Option(6, help="ID de la relation JDM à utiliser")
):
    """Calcule la moyenne incrémentale des similarités intra-type et entraîne un modèle simple."""
    X, y = [], []
    prep_enc, art_enc = build_encoders()

    typer.echo(f"Chargement des corpus depuis {corpus_dir} ...")

    for json_file in tqdm(list(corpus_dir.glob("*.json")), desc="Fichiers corpus"):
        print(json_file)
        label = json_file.stem

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # extraction correcte (structure Corpus)
        inner_data = data.get("data", {})
        relations = [RelationInstance(**v) for v in inner_data.values()]
        print(relations)
        if not relations:
            typer.echo(f"⚠️ Aucun triplet trouvé dans {json_file}")
            continue

        print("Calculating mean resumts..\n")
        # calcul incrémental de la similarité moyenne
        mean_results = incremental_mean_similarity(relations, id_relation, api_call_cache_dir)

        # on ignore le premier triplet (pas de comparaison possible)
        for rel, (simA, simB, score) in zip(relations[1:], mean_results):
            feat = make_feature_vector(rel, simA, simB, score, prep_enc, art_enc)
            X.append(feat)
            y.append(label)

    if not X:
        typer.echo("❌ Aucun vecteur généré : vérifie le contenu des fichiers JSON.")
        raise typer.Exit(code=1)

    X = np.array(X)
    y = np.array(y)
    typer.echo(f"\n✅ Dataset construit : {X.shape[0]} exemples, {X.shape[1]} features")

    # ----------------------
    # Entraînement du modèle
    # ----------------------
    typer.echo("\n🔧 Entraînement du modèle RandomForest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    typer.echo("\n📊 Rapport de classification :\n")
    print(classification_report(y_test, y_pred))


# ---------------------------------------------
# MATRICE DE SIMILARITÉ + CLUSTERING
# ---------------------------------------------

def build_similarity_matrix(relations: list[RelationInstance],
                            id_relation: int,
                            api_call_cache_dir: Path):
    n = len(relations)
    sim_matrix = np.zeros((n, n))
    for (i, t1), (j, t2) in tqdm(combinations(enumerate(relations), 2), total=n*(n-1)//2):
        _, _, score = triplet_similarity(t1, t2, id_relation, api_call_cache_dir)
        sim_matrix[i, j] = sim_matrix[j, i] = score
    return sim_matrix


def cluster_relations(sim_matrix: np.ndarray, k: int = 5):
    dist_matrix = 1 - sim_matrix
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(dist_matrix)
    return labels


def visualize_similarity_matrix(sim_matrix: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, cmap="viridis")
    plt.title("Matrice des similarités entre génitifs")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(labels)), labels, c=labels, cmap="tab10")
    plt.title("Clusters détectés")
    plt.xlabel("Index du triplet")
    plt.ylabel("Cluster")
    plt.show()

def incremental_mean_similarity(relations, id_relation, api_call_cache_dir):
    """Calcule une moyenne approximative de similarités intra-type (linéaire, O(n))"""
    mean_simA, mean_simB, mean_score = 0.0, 0.0, 0.0
    if len(relations) < 2:
        return [(0.0, 0.0, 0.0)]

    results = []
    ref = relations[0]  # point de départ

    for i, rel in tqdm(enumerate(relations[1:], start=2)):
        print(i, rel)
        simA, simB, score = triplet_similarity(rel, ref, id_relation, api_call_cache_dir)

        # moyenne incrémentale
        mean_simA = ((i - 2) * mean_simA + simA) / (i - 1)
        mean_simB = ((i - 2) * mean_simB + simB) / (i - 1)
        mean_score = ((i - 2) * mean_score + score) / (i - 1)

        results.append((mean_simA, mean_simB, mean_score))
        print(results)
        # tu peux mettre à jour la "référence" si tu veux glisser vers les nouveaux
        ref = RelationInstance(
            termA=rel.termA,
            termB=rel.termB,
            prep=rel.prep,
            relation_type=rel.relation_type,
            is_det=rel.is_det,
            determinant=rel.determinant,
        )

    return results


@app.command("matrix-sim")
def matrix_sim_command(
    corpus_dir: Path = typer.Option(..., help="Dossier contenant les fichiers JSON de corpus"),
    api_call_cache_dir: Path = typer.Option(..., help="Dossier cache des appels JDM"),
    id_relation: int = typer.Option(6, help="ID de la relation JDM à utiliser"),
    k: int = typer.Option(5, help="Nombre de clusters KMeans")
):
    """Construit la matrice complète des similarités et effectue un clustering."""
    typer.echo("Chargement des relations...")
    relations = []
    for json_file in corpus_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        inner_data = data.get("data", {})
        relations += [RelationInstance(**v) for v in inner_data.values()]

        typer.echo(f"{len(relations)} triplets chargés.")

        sim_matrix = build_similarity_matrix(relations, id_relation, api_call_cache_dir)
        labels = cluster_relations(sim_matrix, k)
        visualize_similarity_matrix(sim_matrix, labels)

    typer.echo("Analyse terminée ✅")

@app.command("cluster-proto")
def cluster_proto_command(
    corpus_dir: Path = typer.Option(..., "--corpus-dir", help="Dossier des JSON"),
    api_call_cache_dir: Path = typer.Option(..., "--cache-dir", help="Cache JDM"),
    id_relation: int = typer.Option(6, "--rel-id", help="ID relation JDM"),
    k: int = typer.Option(5, "--k", help="Nombre de clusters"),
    show_plot: bool = typer.Option(True, "--plot", help="Afficher graphique")
):
    
    typer.echo("Chargement des prototypes...")
    prototypes = {}
    all_triplets = []

    # 1. Charger un prototype par classe
    for json_file in corpus_dir.glob("*.json"):
        label = json_file.stem
        with open(json_file) as f:
            data = json.load(f)
        relations = [RelationInstance(**v) for v in data.get("data", {}).values()]
        if relations:
            prototypes[label] = relations[0]  # 1er triplet = prototype
            all_triplets.extend([(rel, label) for rel in relations])

    # Afficher une fois, à la fin
    typer.echo(f"{len(prototypes)} prototypes chargés, {len(all_triplets)} triplets au total")

    # 2. Construire les vecteurs de similarité
    X = []
    y_true = []

    for rel, true_label in tqdm(all_triplets, desc="Calcul des features"):
        vec = []
        # Comparer avec TOUS les prototypes
        for proto_label, proto in prototypes.items():  # ← .items() sur le dict
            simA, simB, _ = triplet_similarity(rel, proto, id_relation, api_call_cache_dir)
            vec.append((simA + simB) / 2)
        X.append(vec)
        y_true.append(true_label)

    X = np.array(X)

    k_means = KMeans(n_clusters=k, random_state=11)
    y_pred = k_means.fit_predict(X)

    if show_plot:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='tab10', alpha=0.8)
        plt.title(f"Clustering non supervisé (k={k}) - Prototypes comme références")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        
        # Ajouter les prototypes en gros
        proto_vecs = []
        for proto in prototypes.values():
            vec = []
            for p in prototypes.values():
                simA, simB, _ = triplet_similarity(proto, p, id_relation, api_call_cache_dir)
                vec.append((simA + simB) / 2)
            proto_vecs.append(vec)
        proto_pca = pca.transform(np.array(proto_vecs))
        plt.scatter(proto_pca[:, 0], proto_pca[:, 1], c='red', s=200, marker='X', label='Prototypes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
# ---------------------------------------------
# MAIN
# ---------------------------------------------

if __name__ == "__main__":
    app()
