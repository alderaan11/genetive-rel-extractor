# src/core/visualization/plots.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, classes, figsize=(11, 9), cmap="Blues"):
    """
    Affiche une matrice de confusion lisible.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title("Matrice de confusion")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.show()


def plot_top_features(importances, feature_names, top_k=30, figsize=(10, 8)):
    """
    Affiche un barplot des top features les plus importantes.
    """
    idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in idx]

    plt.figure(figsize=figsize)
    sns.barplot(x=importances[idx], y=names)
    plt.title(f"Top {top_k} Features Importantes")
    plt.tight_layout()
    plt.show()
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_roc_curve(y_true, y_score, classes, figsize=(10, 8)):
    """
    Affiche une courbe ROC multiclasse (One-vs-Rest)
    y_score doit être la probabilité prédite : clf.predict_proba(X_test)
    """
    # Binarisation des labels
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=figsize)

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    # Random guessing
    plt.plot([0, 1], [0, 1], "k--", lw=1)

    plt.title("Courbes ROC - One vs Rest")
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()