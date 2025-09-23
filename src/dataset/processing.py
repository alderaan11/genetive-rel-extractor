#processing corpus.txt 

from pathlib import Path
import json
import typer
from src.models.dataset import Term, Relation, Prep, Corpus

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

def normalize_prep(prep_str: str) -> Prep:
    if prep_str in ("D'un", "D'une", "D'"):
        return Prep.D
    elif prep_str in ("DE", "DES", "DU"):
        return Prep(prep_str)
    else: None

@app.command()
def parse_txt(corpus_path: Path, output_path: Path):


    corpus = Corpus(data=[])

    with open(corpus_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f,1):
            line = line.strip()
            if not line: continue

            example, relation_type = line.split('|')

            relation_type = relation_type.strip()

            parts = example.strip().split(maxsplit=2)
            if len(parts) < 2: continue
            if len(parts) < 3: 
                if parts[1].startswith("D'"):
                    parts.append(parts[1][2:].strip())
                    parts[1] = "D'"

            termA_str, prep_str, termB_str = parts
            print((termA_str, prep_str, termB_str))



            prep_enum = normalize_prep(prep_str)
            

            if prep_enum is None: continue

            if termB_str == "oiseau": 
                print("\noiseau")
                print(termB_str)

            termA = Term(name=termA_str)
            termB = Term(name=termB_str)


            relation = Relation(termA=termA, termB=termB, prep=prep_enum, rel_type=relation_type)

            corpus.data.append(relation)

    with open(output_path, "w+", encoding="utf-8") as f:
        json.dump(
            [d.model_dump() for d in corpus.data], f, ensure_ascii=False, indent=2
        )

        typer.echo(f"{len(corpus.data)} relations parsée et sauvegardées dans {output_path}")

if __name__ == '__main__':
    app()