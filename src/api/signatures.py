import requests
import json
from pathlib import Path
from typing import List, Optional
import typer
from src.models.dataset import Term, Relation

app = typer.Typer()

def get_infos_by_name(term: str):
    # url = f"https://jdm-api.demo.lirmm.fr/v0/node_by_name/{term}"
    url = f"https://jdm-api.demo.lirmm.fr/v0/refinements/{term}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        res = response.json()
        print(res)
        # return res["type"]
    except requests.exceptions.RequestException as e:
        typer.echo(f"Erreur lors de la requête : {e}")
        return None


def get_infos_by_id(idB: int):
    url = f"https://jdm-api.demo.lirmm.fr/v0/node_by_id/{idB}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        res = response.json()
        return res["name"]
    except requests.exceptions.RequestException as e:
        typer.echo(f"Erreur lors de la requête : {e}")
        return None


def get_relation_name(rel: List, id_rel: int):
    for item in rel:
        if item["id"] == id_rel:
            return item["nom"]
    return None


def get_relations_from_name(term: str):
    url = f"https://jdm-api.demo.lirmm.fr/v0/relations/from/{term}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        res = response.json()
        return res["relations"]
    except requests.exceptions.RequestException as e:
        typer.echo(f"Erreur lors de la requête : {e}")
        return None


@app.command()
def explore(term: str, top: int = 11):
    with open("data/relations.json") as f:
        rel_list = json.load(f)

    cache_path = Path(f"cache/{term}.json")

    sign = get_infos_by_name(term)
    print(f"SIGN : {sign}")

    if not cache_path.exists():
        signature = get_relations_from_name(term)
        if not signature:
            typer.echo("Aucune relation trouvée.")
            raise typer.Exit()

        sorted_signature = sorted(signature, key=lambda x: x["w"], reverse=True)

        results = []
        for item in sorted_signature[:top]:
            name_b = get_infos_by_id(item["node2"])
            if name_b and not name_b.startswith("_") and not name_b.startswith(":"):
                rel_name = get_relation_name(rel_list, item["type"])
                results.append((name_b, rel_name))

        for name_b, rel_name in results:
            typer.echo(f"{term} --[{rel_name}]--> {name_b}")

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(sorted_signature, f, ensure_ascii=False, indent=2)
    else:
        typer.echo(f"Cache déjà présent : {cache_path}")


if __name__ == "__main__":
    app()
