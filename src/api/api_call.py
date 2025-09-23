from pathlib import Path
import json
import typer
from src.models.dataset import Term, Relation, Prep, Corpus
import requests
from typing import List, Dict

app = typer.Typer()


def get_infos_by_name(term: str) -> Dict:
    # url = f"https://jdm-api.demo.lirmm.fr/v0/refinements/{term}"
    url = f"https://jdm-api.demo.lirmm.fr/v0/node_by_name/{term}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()        
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

def get_relation_name(rel: List, id_rel: int):
    for item in rel:
        if item["id"] == id_rel:
            return item["nom"]
    return None


@app.command()
def explore(corpus_path: Path, top: int = 5):
    with open(corpus_path) as f: corpus = json.load(f)


    for example in corpus:
        terms = [
            Term(name=example["termA"]["name"]),
            Term(name=example["termB"]["name"])
        ]
        
        for term_obj in terms:
            t_name = term_obj.name
            print(t_name)
            cache_path = Path(f"cache/{t_name}.json")
            if not cache_path.exists():
                raffinement = get_relations_from_name(t_name)
                raff = []

                if raffinement:
                    # Filtrer type==6
                    type6 = [x for x in raffinement if x["type"] == 6]

                    raff_temp = []
                    for x in type6:
                        node2_name = get_infos_by_id(x["node2"])
                        if not node2_name.startswith("_") and ">" not in node2_name:
                            raff_temp.append({"node2": node2_name, "w": x["w"]})

                    # Trier et limiter à 5
                    raff_sorted = sorted(raff_temp, key=lambda d: d["w"], reverse=True)[:5]

                    # Ajouter les noms à hypernym de Term
                    term_obj.hypernym.update({d["node2"]: d["w"] for d in raff_sorted})

                print(f"Raffinement: {term_obj.hypernym}\n")

        example["termA"]["hypernym"] = terms[0].hypernym
        example["termB"]["hypernym"] = terms[1].hypernym

    with open(corpus_path, "w") as f:
        json.dump(corpus, f, ensure_Ascii=False, indent=2)

if __name__ == '__main__':
    app()
            

            

        