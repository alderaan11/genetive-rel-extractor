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
def parse_txt(corpus_dir: Path, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = set()


    for txt_file in corpus_dir.glob("*.txt"):
        corpus_path = txt_file
        output_path = output_dir / (txt_file.stem + ".json")    

        corpus = Corpus(data=[])

        with open(corpus_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Séparer exemple et type de relation
                try:
                    example, relation_type = line.split('|')
                except ValueError:
                    print(f"⚠️ Ligne {lineno} mal formée : {line}")
                    continue

                relation_type = relation_type.strip()

                # Découper en [termA, prep, termB]
                parts = example.strip().split(maxsplit=2)
                if len(parts) < 3:
                    continue

                termA_str, prep_str, termB_str = parts

                # Normaliser la préposition
                prep_enum = normalize_prep(prep_str)
                if prep_enum is None:
                    continue

                DET = False
                if (relation_type == "r_own-1"): 
                    termB_strcopy = termB_str.lower()
                if prep_enum in ["DE", "DU", "DES", "D'"]:
                    if termB_str.startswith(("le ", "la ", "les ", "l'")):
                        DET = True
                        if termB_str.startswith("l'"):
                            termB_str = termB_str[2:].strip()
                        else:
                            termB_str = termB_str.split(maxsplit=1)[1]
                
                if DET == True and relation_type == "r_own-1":
                    print(termB_strcopy)
                    

                # Création des termes
                termA = Term(name=termA_str.lower())
                termB = Term(name=termB_str.lower())

                # Relation avec info sur DET
                relation = Relation(
                    termA=termA,
                    termB=termB,
                    prep=prep_enum,
                    rel_type=relation_type,
                    det=DET  # ajout du booléen
                )

                corpus.data.append(relation)

                vocabulary.add(termA.name)
                vocabulary.add(termB.name)
            

        vocabulary_path = Path("data/vocabulary.json")
        vocabulary_list = [
            {"name": term, "r_isa": [], "r_raff_sem": [], "r_pos": []} for term in sorted(vocabulary)
        ]

        # Sauvegarde si besoin
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump([rel.model_dump() for rel in corpus.data], f_out, ensure_ascii=False, indent=2)

        with open(vocabulary_path, "w", encoding="utf-8") as f_vocab:
            json.dump(vocabulary_list, f_vocab, ensure_ascii=False, indent=2)
        print(f"Fichier traité : {txt_file} -> {output_path}")

if __name__ == '__main__':
    app()