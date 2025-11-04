import numpy as np
import pickle
import typer
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import json
from src.models.dataset import GenitiveDataset

app = typer.Typer()

class Trainer:
    def __init__(self, n_components=150, classifier_type="logistic_regression"):
        self.n_components = n_components
        self.classifier_type = classifier_type
        self.scaler = None
        self.svd = None
        self.classifier = None
        self.vocab_info = None

    def apply_dim_reduction(self, X: np.ndarray, fit=True) -> np.ndarray:
        if fit:
            n_components = min(self.n_components, X.shape[0] - 1, X.shape[1])
            self.svd = TruncatedSVD(n_components=n_components, random_state=11)
            X_reduced = self.svd.fit_transform(X)

            explained_var = self.svd.explained_variance_ratio_.sum()
            
            print(f"\n  Variance par composante (top 10):")
            for i, var in enumerate(self.svd.explained_variance_ratio_[:10], 1):
                bar = '█' * int(var * 200)
                print(f"    PC{i:2d}: {var:.4f} {bar}")

            self.scaler = RobustScaler(quantile_range=(10,90))
            X_reduced = self.scaler.fit_transform(X_reduced)
        
        else:
            X_reduced = self.svd.transform(X)
            X_reduced = self.scaler.transform(X_reduced)

        return X_reduced

    def create_classifier(self):
        if self.classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(max_iter=1000, random_state=11)
        elif self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=11)
        elif self.classifier_type == "svm":
            self.classifier = LinearSVC(max_iter=1000, random_state=11)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def cross_validate(self, X: np.ndarray, y: np.ndarray, nb_fold: int = 5) -> None:
        self.classifier = self.create_classifier()
        cv = StratifiedKFold(n_splits=nb_fold, shuffle=True, random_state=11)
        metrics = {
            'f1_macro': 'F1_Score (macro)',
            'accuracy': 'Accuracy',
            'precision_macro': 'Precision (macro)',
            'recall_macro': 'Recall (macro)',
            'f1_weighted': 'F1_Score (weighted)'
        }

        results = {}

        for metric_name, metric_label in metrics.items():
            scores = cross_val_score(self.classifier, X, y, cv=cv, scoring=metric_name, n_jobs=-1)
            results[metric_label] = scores
            print(f"\n{metric_label}:")
            print(f"  Folds: {', '.join([f'{s:.4f}' for s in scores])}")
            print(f"  Moyenne: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return results

    def train(self, X: np.ndarray, y: np.ndarray):
        self.classifier = self.create_classifier()
        self.classifier.fit(X,y)

        y_pred = self.classifier.predict(X)

        n_classes = len(np.unique(y))

        if n_classes <= 10:
            print("\nMatrice de confusion:")
            cm = confusion_matrix(y, y_pred)
            classes = sorted(np.unique(y))
            
            # Header
            print(f"\n{'':20s}", end='')
            for cls in classes:
                print(f"{cls[:8]:>8s}", end='')
            print()
            
            # Lignes
            for i, cls in enumerate(classes):
                print(f"{cls:20s}", end='')
                for j in range(len(classes)):
                    val = cm[i, j]
                    if val > 0:
                        print(f"{val:8d}", end='')
                    else:
                        print(f"{'·':>8s}", end='')
                print()
        
        return self.classifier
    
    def save_model(self, output_path: str, vocab_info: dict):
        """Sauvegarde le modèle complet"""
        model_data = {
            'vocab_info': vocab_info,
            'svd': self.svd,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'n_components': self.n_components
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Modèle sauvegardé: {output_path}")
        print(f"  Taille: {Path(output_path).stat().st_size / 1024:.2f} KB")
    
    def save_metrics(self, cv_results: dict, output_path: str):
        """Sauvegarde les métriques de validation croisée"""
        metrics_data = {
            'classifier_type': self.classifier_type,
            'n_components': self.n_components,
            'cv_results': {
                metric: {
                    'scores': scores.tolist(),
                    'mean': float(scores.mean()),
                    'std': float(scores.std())
                }
                for metric, scores in cv_results.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Métriques sauvegardées: {output_path}")

@app.command()
def main(
    data: str = typer.Argument(..., help="Chemin vers le fichier JSON des données"),
    transform: str = typer.Option("log1p", help="Transformation des poids", autocompletion=lambda: ["log1p", "sqrt", "boxcox", "rank", "none"]),
    n_components: int = typer.Option(150, help="Nombre de composantes SVD"),
    classifier: str = typer.Option("logistic_regression", help="Type de classifieur", autocompletion=lambda: ["logistic_regression", "logistic_l1", "rf", "svm"]),
    no_det: bool = typer.Option(False, help="Ne pas utiliser la feature déterminant"),
    no_prep: bool = typer.Option(False, help="Ne pas utiliser la feature préposition"),
    output: str = typer.Option("model.pkl", help="Chemin de sortie du modèle"),
    cv_folds: int = typer.Option(5, help="Nombre de folds pour validation croisée")
):
    print("="*60)
    print("ENTRAÎNEMENT - CLASSIFICATION RELATIONS GÉNITIVES")
    print("="*60)
    print(f"\nParamètres:")
    print(f"  Données: {data}")
    print(f"  Transform: {transform}")
    print(f"  Composantes SVD: {n_components}")
    print(f"  Classifieur: {classifier}")
    print(f"  Feature déterminant: {not no_det}")
    print(f"  Feature préposition: {not no_prep}")
    
    # 1. Préparer le dataset
    dataset = GenitiveDataset(
        transform=transform,
        use_det_feature=not no_det,
        use_prep_feature=not no_prep
    )
    
    dataset.load_data(data)
    dataset.build_vocabulary()
    X, y = dataset.prepare_dataset()
    vocab_info = dataset.get_vocab_info()
    
    # 2. Créer le trainer
    trainer = Trainer(
        n_components=n_components,
        classifier_type=classifier
    )
    
    # 3. Réduction de dimensionnalité
    X_reduced = trainer.apply_dim_reduction(X, fit=True)
    
    # 4. Validation croisée
    cv_results = trainer.cross_validate(X_reduced, y, nb_fold=cv_folds)
    
    # 5. Entraînement final
    trainer.train(X_reduced, y)
    
    # 6. Sauvegarder
    trainer.save_model(output, vocab_info)
    
    metrics_path = output.replace('.pkl', '_metrics.json')
    trainer.save_metrics(cv_results, metrics_path)
    
    print("\n" + "="*60)
    print("✓ ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print(f"\nFichiers créés:")
    print(f"  - Modèle: {output}")
    print(f"  - Métriques: {metrics_path}")

if __name__ == "__main__":
    app()