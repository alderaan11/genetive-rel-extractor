from pathlib import Path
import json
from typing import Dict, Tuple
import numpy as np
import typer

app = typer.Typer()


def load_features_from_dir(features_dir: Path):
    X_all, y_all = [], []
    rule_ids = None

    for feat_file in features_dir.glob("*_features.json"):
        with open(feat_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        X_all.extend(data["X"])
        y_all.extend(data["y"])

        if rule_ids is None:
            rule_ids = data.get("rule_ids") or data.get("rule_labels")

    return np.array(X_all), np.array(y_all), rule_ids


def load_rules_counts(rules_dir: Path) -> Dict[str, int]:
    counts = {}
    # TRI ALPHABÉTIQUE STRICT
    for rule_file in sorted(rules_dir.glob("*_rules.json"), key=lambda p: p.name):
        label = rule_file.stem.replace("_rules", "")
        with open(rule_file, "r", encoding="utf-8") as f:
            rules = json.load(f)
        counts[label] = len(rules)
    return counts


def build_feature_ranges(counts: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    ranges = {}
    start = 0
    for label, size in counts.items():
        end = start + size - 1
        ranges[label] = (start, end)
        start = end + 1
    return ranges


@app.command()
def explain_feature(
    index: int = typer.Argument(..., help="Index global de la feature"),
    features_dir: Path = typer.Option(..., "--features-dir"),
    rules_dir: Path = typer.Option(..., "--rules-dir"),
):
    X, y, rule_ids = load_features_from_dir(features_dir)

    counts = load_rules_counts(rules_dir)
    ranges = build_feature_ranges(counts)

    typer.echo(f"Nombre total de règles concaténées (alphabétique) : {sum(counts.values())}")
    typer.echo(f"Index demandé : {index}\n")

    sorted_rule_files = sorted(rules_dir.glob("*_rules.json"), key=lambda p: p.name)

    # Recherche du bon type génitif
    for rule_file in sorted_rule_files:
        label = rule_file.stem.replace("_rules", "")
        start, end = ranges[label]

        if start <= index <= end:
            local_index = index - start

            with open(rule_file, "r", encoding="utf-8") as f:
                rules = json.load(f)

            rule = rules[local_index]

            typer.echo(f"📌 Feature #{index} correspond à :")
            typer.echo(f"   → Type génitif : {label}")
            typer.echo(f"   → Index local  : {local_index} (plage {start}–{end})")
            typer.echo(f"   → Fichier      : {rule_file}")

            typer.echo("\n📜 Contenu de la règle :\n")
            typer.echo(json.dumps(rule, indent=2, ensure_ascii=False))
            return

    typer.echo("❌ Index en dehors des plages de règles.")


if __name__ == "__main__":
    app()
