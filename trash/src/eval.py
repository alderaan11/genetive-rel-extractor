"""
eval.py - Évaluation et prédiction avec le modèle entraîné
"""

import numpy as np
import pickle
import argparse
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import GenitiveDataset, Corpus, Relation, Term, Prep, RelationItem


class GenitivePredictor:
    """Classe pour faire des prédictions avec un modèle entraîné"""
    
    def __init__(self, model_path: str):
        """Charge un modèle sauvegardé"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_info = model_data['vocab_info']
        self.svd = model_data['svd']
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.classifier_type = model_data['classifier_type']
        self.n_components = model_data['n_components']
        
        print(f"✓ Modèle chargé: {model_path}")
        print(f"  Type: {self.classifier_type}")
        print(f"  Composantes: {self.n_components}")
    
    def prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Applique les transformations (SVD + scaler)"""
        X_reduced = self.svd.transform(X)
        X_scaled = self.scaler.transform(X_reduced)
        return X_scaled
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les classes"""
        X_transformed = self.prepare_features(X)
        return self.classifier.predict(X_transformed)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prédit les probabilités (si disponible)"""
        X_transformed = self.prepare_features(X)
        
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X_transformed)
        elif hasattr(self.classifier, 'decision_function'):
            # Pour SVM
            scores = self.classifier.decision_function(X_transformed)
            if len(scores.shape) == 1:
                # Binary classification
                return np.vstack([1 - scores, scores]).T
            # Normaliser avec softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        else:
            raise ValueError("Le classifieur ne supporte pas predict_proba")
    
    def predict_single(self, relation: Relation, dataset: GenitiveDataset) -> Tuple[str, dict]:
        """Prédit une seule relation"""
        features = dataset.relation_to_features(relation).reshape(1, -1)
        prediction = self.predict(features)[0]
        
        result = {'prediction': prediction}
        
        # Ajouter les probabilités si disponible
        try:
            probas = self.predict_proba(features)[0]
            classes = self.classifier.classes_
            result['probabilities'] = {cls: float(prob) for cls, prob in zip(classes, probas)}
            result['top_3'] = sorted(
                zip(classes, probas), 
                key=lambda x: -x[1]
            )[:3]
        except:
            pass
        
        return prediction, result


class GenitiveEvaluator:
    """Classe pour évaluer le modèle sur un jeu de test"""
    
    def __init__(self, predictor: GenitivePredictor):
        self.predictor = predictor
        self.y_true = None
        self.y_pred = None
        self.classes = None
    
    def evaluate(self, dataset: GenitiveDataset, verbose=True) -> Dict:
        """Évalue le modèle sur un dataset"""
        X, y = dataset.get_data()
        
        # Prédictions
        self.y_true = y
        self.y_pred = self.predictor.predict(X)
        self.classes = sorted(np.unique(y))
        
        # Calcul des métriques
        metrics = {
            'accuracy': (self.y_pred == self.y_true).mean(),
            'f1_macro': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'f1_per_class': {}
        }
        
        # F1 par classe
        for cls in self.classes:
            y_true_binary = (self.y_true == cls).astype(int)
            y_pred_binary = (self.y_pred == cls).astype(int)
            metrics['f1_per_class'][cls] = f1_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
        
        if verbose:
            self.print_report()
        
        return metrics
    
    def print_report(self):
        """Affiche un rapport détaillé"""
        print("\n" + "="*60)
        print("RAPPORT D'ÉVALUATION")
        print("="*60)
        
        print(f"\n✓ Accuracy: {(self.y_pred == self.y_true).mean():.4f}")
        print(f"✓ F1-Score (macro): {f1_score(self.y_true, self.y_pred, average='macro', zero_division=0):.4f}")
        print(f"✓ F1-Score (weighted): {f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0):.4f}")
        
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60)
        print(classification_report(self.y_true, self.y_pred, zero_division=0))
    
    def plot_confusion_matrix(self, output_path: Path = None, normalize=False):
        """Affiche/sauvegarde la matrice de confusion"""
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Matrice de confusion (normalisée)'
        else:
            fmt = 'd'
            title = 'Matrice de confusion'
        
        # Limiter à 15 classes max pour la lisibilité
        if len(self.classes) > 15:
            print("⚠️  Plus de 15 classes, affichage simplifié")
            # Afficher seulement le top 15
            class_counts = [(cls, (self.y_true == cls).sum()) for cls in self.classes]
            top_classes = [cls for cls, _ in sorted(class_counts, key=lambda x: -x[1])[:15]]
            
            mask = np.isin(self.y_true, top_classes) & np.isin(self.y_pred, top_classes)
            y_true_filtered = self.y_true[mask]
            y_pred_filtered = self.y_pred[mask]
            cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
            classes = top_classes
            title += ' (top 15 classes)'
        else:
            classes = self.classes
        
        plt.figure(figsize=(max(10, len(classes)), max(8, len(classes) * 0.8)))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Nombre de prédictions'})
        plt.title(title)
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Matrice de confusion sauvegardée: {output_path}")
        else:
            plt.show()
    
    def analyze_errors(self, top_n=10):
        """Analyse les erreurs les plus fréquentes"""
        print("\n" + "="*60)
        print(f"TOP {top_n} ERREURS LES PLUS FRÉQUENTES")
        print("="*60)
        
        errors = []
        for true_cls, pred_cls in zip(self.y_true, self.y_pred):
            if true_cls != pred_cls:
                errors.append((true_cls, pred_cls))
        
        if not errors:
            print("✓ Aucune erreur!")
            return
        
        from collections import Counter
        error_counts = Counter(errors)
        
        print(f"\nTotal d'erreurs: {len(errors)} / {len(self.y_true)} ({len(errors)/len(self.y_true)*100:.1f}%)\n")
        
        for i, ((true_cls, pred_cls), count) in enumerate(error_counts.most_common(top_n), 1):
            percentage = count / len(errors) * 100
            print(f"{i:2d}. {true_cls:20s} → {pred_cls:20s} : {count:3d} fois ({percentage:.1f}%)")
    
    def get_misclassified_examples(self, dataset: GenitiveDataset, limit=5) -> List[Dict]:
        """Retourne des exemples mal classifiés"""
        misclassified = []
        
        for i, (true_cls, pred_cls) in enumerate(zip(self.y_true, self.y_pred)):
            if true_cls != pred_cls and len(misclassified) < limit:
                relation = dataset.corpus.data[i]
                misclassified.append({
                    'index': i,
                    'termA': relation.termA.name,
                    'termB': relation.termB.name,
                    'prep': relation.prep.value,
                    'det': relation.det,
                    'true_class': true_cls,
                    'predicted_class': pred_cls
                })
        
        return misclassified


def main():
    parser = argparse.ArgumentParser(description='Évaluer le classifieur de relations génitives')
    parser.add_argument('--model', type=str, required=True,
                       help='Chemin vers le modèle .pkl')
    parser.add_argument('--data', type=str, required=True,
                       help='Chemin vers le fichier JSON de test')
    parser.add_argument('--transform', type=str, default='log1p',
                       help='Transformation des poids (doit correspondre au modèle)')
    parser.add_argument('--confusion_matrix', type=str, default=None,
                       help='Chemin pour sauvegarder la matrice de confusion')
    parser.add_argument('--normalize_cm', action='store_true',
                       help='Normaliser la matrice de confusion')
    parser.add_argument('--show_errors', type=int, default=10,
                       help='Nombre d\'erreurs à afficher')
    parser.add_argument('--show_examples', type=int, default=5,
                       help='Nombre d\'exemples mal classifiés à afficher')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ÉVALUATION - CLASSIFICATION RELATIONS GÉNITIVES")
    print("="*60)
    
    # 1. Charger le modèle
    predictor = GenitivePredictor(args.model)
    
    # 2. Charger les données de test avec les mêmes paramètres
    dataset = GenitiveDataset(
        transform=args.transform,
        use_det_feature=predictor.vocab_info['use_det_feature'],
        use_prep_feature=predictor.vocab_info['use_prep_feature']
    )
    
    # Utiliser le vocabulaire du modèle
    dataset.load_data(args.data)
    dataset.vocab = predictor.vocab_info['vocab']
    dataset.prep_to_idx = predictor.vocab_info['prep_to_idx']
    
    # Préparer le dataset
    X, y = dataset.prepare_dataset()
    
    # 3. Évaluer
    evaluator = GenitiveEvaluator(predictor)
    metrics = evaluator.evaluate(dataset)
    
    # 4. Analyser les erreurs
    if args.show_errors > 0:
        evaluator.analyze_errors(top_n=args.show_errors)
    
    # 5. Exemples mal classifiés
    if args.show_examples > 0:
        print("\n" + "="*60)
        print(f"EXEMPLES MAL CLASSIFIÉS (top {args.show_examples})")
        print("="*60)
        
        examples = evaluator.get_misclassified_examples(dataset, limit=args.show_examples)
        for ex in examples:
            print(f"\n#{ex['index']}: {ex['termA']} {ex['prep']} {ex['termB']}")
            print(f"  Vraie classe: {ex['true_class']}")
            print(f"  Prédiction:   {ex['predicted_class']}")
            print(f"  Déterminant:  {ex['det']}")
    
    # 6. Matrice de confusion
    if args.confusion_matrix:
        pass
        # output_path = Path(args.confusion_matrix)
        # evaluator.plot_confusion_matrix(output_path, normalize