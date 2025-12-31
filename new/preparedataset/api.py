import requests
import json
from pathlib import Path
from typing import List, Optional
import typer
from tqdm import tqdm
from ..schemas.base_models import TermInfo, RelationInstance, Corpus, Node, ApiCall
from ..utils.logger import logger
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



def explore(term: str, id_relation: int, output_dir=Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    relation_file = output_dir / f"infos_by_name_{id_relation}.json"

    if relation_file.exists():
        with open(relation_file, "r", encoding="utf-8") as f:
            apicall_data = json.load(f)
            api_call = ApiCall(**apicall_data)

    else: api_call = ApiCall(id_relation=id_relation, relation_nodes={})

    if term in api_call.relation_nodes:
        print(api_call.relation_nodes[term])
        return

    sign = get_infos_by_name(term)
    print(f"SIGN : {sign}")

    result = get_relations_from_name(term)

    if not result:
        typer.echo("Aucune relation trouvée.")
        return

    sorted_result = sorted(result, key=lambda x: x["w"], reverse=True)
    filtered_result = [x for x in sorted_result if x["type"] == id_relation]
    
    if not filtered_result: 
        print("Error while printing")
        return 

    logger.info(f"Filtered API call result : {filtered_result}")
    nodes = [Node(id_node=item["id"],node1=item["node1"], node2=item["node2"], weight=item["w"]) for item in filtered_result]

    api_call.relation_nodes[term] = nodes

    with open(relation_file, "w", encoding="utf-8") as f:
        f.write(api_call.model_dump_json(indent=2))

    
@app.callback(invoke_without_command=True)
def fetch_vocabulary_by_id(
    vocab_path: Path = typer.Option(..., "--vocab-path"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    id_relation: List[int] = typer.Option(..., "--id-relation"),
    delay: float = typer.Option(0.5),
):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    for word in tqdm(vocabulary):
        for id_rel in id_relation:
            explore(word, id_rel, output_dir)



if __name__ == "__main__":
    app()