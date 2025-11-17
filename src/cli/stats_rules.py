import json
from pathlib import Path
from typing import Dict, List
from collections import Counter
import numpy as np
import typer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

app = typer.Typer(help="Analyse statistique + visualisations sur les règles génitives")

sns.set_theme(style="whitegrid")


# ------------------------
# Chargement des règles
# ------------------------
def load_rules(rules_dir: Path) -> Dict[str, List[dict]]:
    rules = {}
    for rule_file in sorted(rules_dir.glob("*_rules.json"), key=lambda p: p.name):
        label = rule_file.stem.replace("_rules", "")
        with open(rule_file, "r", encoding="utf-8") as f:
            rules[label] = json.load(f)
    return rules


# ---------------------------
# Statistiques de base
# ---------------------------
def collect_node_stats(rule_list: List[dict]):
    counter_a, counter_b = Counter(), Counter()

    for r in rule_list:
        for k, v in r["nodes_a"].items():
            counter_a[k] += abs(v)
        for k, v in r["nodes_b"].items():
            counter_b[k] += abs(v)

    return counter_a, counter_b


def build_matrix(rule_list: List[dict], all_nodes: List[str], side="nodes_a"):
    mat = np.zeros((len(rule_list), len(all_nodes)))
    for i, r in enumerate(rule_list):
        for j, n in enumerate(all_nodes):
            if n in r[side]:
                mat[i, j] = r[side][n]
    return mat


# -------------------------
# Commande principale
# -------------------------
@app.command()
def analyze(
    rules_dir: Path = typer.Option(..., "--rules-dir"),
    export: Path = typer.Option("rules_stats.json", "--export")
):
    typer.echo("Chargement des règles...")
    rules = load_rules(rules_dir)

    stats = {}
    global_counter = Counter()

    # =======================
    # Analyse par type
    for label, rule_list in rules.items():
        typer.echo(f"\nAnalyse {label} ({len(rule_list)} règles)...")

        counter_a, counter_b = collect_node_stats(rule_list)

        # accumulate global stats
        global_counter.update(counter_a)
        global_counter.update(counter_b)

        top_a = counter_a.most_common(3)
        top_b = counter_b.most_common(3)

        stats[label] = {
            "nb_rules": len(rule_list),
            "top_nodes_a": top_a,
            "top_nodes_b": top_b,
        }

        
        if top_a:
            df = pd.DataFrame(top_a, columns=["node", "weight"])
            plt.figure(figsize=(5, 2.5))
            sns.barplot(x="weight", y="node", data=df, palette="Blues_r")
            plt.title(f"Top nodes A – {label}")
            plt.show()

        if top_b:
            df = pd.DataFrame(top_b, columns=["node", "weight"])
            plt.figure(figsize=(5, 2.5))
            sns.barplot(x="weight", y="node", data=df, palette="Greens_r")
            plt.title(f"Top nodes B – {label}")
            plt.show()


    # =======================
    # Top 10 global
    # =======================
    top_global = global_counter.most_common(10)

    typer.echo("\nTop 10 nodes globaux :")
    for nid, val in top_global:
        typer.echo(f"  {nid:>10} → {val:.2f}")

    plt.figure(figsize=(8, 4))
    df_global = pd.DataFrame(top_global, columns=["node", "weight"])
    sns.barplot(x="weight", y="node", data=df_global, palette="rocket")
    plt.title("Top 10 nodes globaux")
    plt.show()

    # =======================
    # Heatmap "présence"
    # =======================
    presence = {
        label: {
            nid: any(nid in r["nodes_a"] or nid in r["nodes_b"] for r in rule_list)
            for nid, _ in top_global
        }
        for label, rule_list in rules.items()
    }

    df_presence = pd.DataFrame(presence).T.astype(int)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_presence, annot=True, cmap="YlGnBu")
    plt.title("Présence des Top-10 globaux dans chaque type de génitif")
    plt.xlabel("Node ID")
    plt.ylabel("Relation génitive")
    plt.show()

    # =======================
    # Export JSON
    # =======================
    with open(export, "w", encoding="utf-8") as f:
        json.dump({
            "top_global": top_global,
            "stats": stats,
            "presence_top10": df_presence.to_dict()
        }, f, indent=2, ensure_ascii=False)

    typer.echo(f"\nExport sauvegardé → {export}")


if __name__ == "__main__":
    app()
