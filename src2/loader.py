from pathlib import Path
import json
import typer
from src2.base import TermInfo, RelationInstance, Prep, Corpus, Article
from typing import List, Dict, Any
import re
app = typer.Typer()

PREP_MAP = {
    "DE": Prep.DE,
    "DU": Prep.DU,
    "DES": Prep.DES,
    "D'": Prep.D,
    "D": Prep.D,
    "Dun": Prep.D,
    "D'une": Prep.D,
    "D'un": Prep.D,
}

ARTICLES = {'le', 'la', 'les', "l'"}

@app.command()
def parse_line(line: str) -> Dict[str, Any]:
    relation_instance = {
        "termA": "",
        "termB": "",
        "prep": "",
        "determinant": None,
        "is_det": False,
        "relation_type": ""
    }

    line_stripped = line.strip()
    if not line_stripped:
        return {}

    instance, relation_type = line_stripped.split('|')
    relation_instance["relation_type"] = relation_type.strip()

    instance_splitted = re.findall(r"\b\w+'\b|\b\w+\b", instance)

    if not instance_splitted:
        return relation_instance

    relation_instance["termA"] = instance_splitted[0]

    prep_value = {p.value: p for p in Prep}

    for i in range(len(instance_splitted) - 1, -1, -1):
        current_token = instance_splitted[i].upper()
        if current_token in prep_value:
            prep = prep_value[current_token]
            relation_instance["prep"] = prep

            if prep.value in ["DE", "DES", "DU", "D'"]:
                if i + 1 < len(instance_splitted):
                    first = instance_splitted[i + 1]
                    termB = first

                    if first.lower() in ARTICLES:
                        if i + 2 < len(instance_splitted):
                            second = instance_splitted[i + 2]
                            termB = second

                        relation_instance["is_det"] = True
                        relation_instance["determinant"] = Article(first.upper())

                    relation_instance["termB"] = termB



            else:
                if i + 2 < len(instance_splitted):
                    termB = instance_splitted[i + 1] + " " + instance_splitted[i + 2]
                    relation_instance["termB"] = termB
                    relation_instance["is_det"] = True

            break
    print(relation_instance)
    return relation_instance

@app.command()
def parse_txt_file(txt_path: Path):
    corpus = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            instance = line.split('|')[0]
            parsed_line = parse_line(line)
            if not parsed_line.get("prep") or not parsed_line.get("termB"):
                continue
            rel = RelationInstance(
                termA=TermInfo(name=parsed_line["termA"].lower()),
                termB=TermInfo(name=parsed_line["termB"]),
                prep=parsed_line["prep"],
                relation_type=parsed_line["relation_type"],
                is_det=parsed_line["is_det"],
                determinant=parsed_line["determinant"]
            )
            corpus[instance] = rel


    return corpus

@app.command()
def parse_txt_dir(input_dir: Path, output_dir: Path):
    vocabulary = set()

    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in input_dir.glob("*.txt"):
        output_path  = output_dir / (txt_file.stem + ".json")

        corpus = Corpus(original_file = Path(txt_file), data=parse_txt_file(txt_file))

        for _, rel in corpus.data.items():
            vocabulary.update({rel.termA.name, rel.termB.name})


        with open(output_path, "w", encoding="utf-8") as f:
            f.write(corpus.model_dump_json(indent=2))

    typer.echo(f"Parsing finished for {output_dir}\n")
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(sorted(vocabulary), f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    app()


