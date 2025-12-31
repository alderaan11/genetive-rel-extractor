from pathlib import Path
import json
import typer
from ..schemas.base_models import TermInfo, RelationInstance, Prep, Corpus, Article
from typing import List, Dict, Any
import re
from loguru import logger
from ..utils.logger import logger
app = typer.Typer()

PREP_MAP = {
    "DE": Prep.DE,
    "DE la": Prep.DE,
    "DE l'": Prep.DE,
    "DU": Prep.DU,
    "DES": Prep.DES,
    "D'": Prep.D,
    "D '": Prep.D,
    "D": Prep.D,
    "Dun": Prep.D,
    "D'une": Prep.D,
    "D'un": Prep.D,
}

PREPOSITION = {"DE", "DES", "DU", "D'"}

ARTICLES = {'le', 'la', 'les', "l'", "un", "une"}

def parse_line(line: str) -> RelationInstance:
    line_stripped = line.strip()
    relation_instance = RelationInstance()
    instance, relation_type = line_stripped.split('|')

    instance_splitted = re.findall(r"\b\w+'\b|\b\w+\b", instance)

    relation_instance.termA = TermInfo(name=instance_splitted[0])

    prep_value = {p.value: p for p in Prep}
    for i in range(len(instance_splitted) - 1, -1, -1):
        current_token = instance_splitted[i].upper()
        if current_token in prep_value:
            prep = prep_value[current_token]
            relation_instance.prep = prep
        
            if relation_instance.prep and relation_instance.prep.value in PREPOSITION:
                if i + 1 < len(instance_splitted):
                    first = instance_splitted[i+1]
                    termB = first

                    if first.lower() in ARTICLES:
                        if i+2 < len(instance_splitted):
                            second = instance_splitted[i+2]
                            termB = second

                        relation_instance.is_det = True
                        relation_instance.determinant = Article(first.upper())
                    relation_instance.termB = TermInfo(name=termB)

            else:
                if i + 2 < len(instance_splitted):
                    termB = instance_splitted[i+1] + " " + instance_splitted[i+2]
                    relation_instance.termB = termB
                    relation_instance.is_det = False
                break

    return relation_instance
def parse_txt_file(txt_path: Path):
    corpus = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            logger.info(f"To be parsed : {line}")
            instance = line.split('|')[0]
            parsed_line = parse_line(line)

            if not parsed_line.prep or not parsed_line.termB:
                continue
            parsed_line.relation_type = line.split('|')[1]
            logger.info(f"Parsed : {parsed_line}")

            corpus[instance] = parsed_line

    return corpus

@app.callback(invoke_without_command=True)
def main(
    txt_sample_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
):
    vocabulary = set()

    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in txt_sample_dir.glob("*.txt"):
        output_path = output_dir / (txt_file.stem + ".json")
        corpus = Corpus(original_file=Path(txt_file), data=parse_txt_file(txt_file))

        for _, rel in corpus.data.items():
            vocabulary.update({rel.termA.name, rel.termB.name})

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(corpus.model_dump_json(indent=2))

    typer.echo(f"Parsing finished for {output_dir}\n")
    vocab_path = output_dir.parents[0] / "vocabulary.json"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(sorted(vocabulary), f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    app()