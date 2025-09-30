from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics
import typer
from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt


app = typer.Typer()

class RelationItem(BaseModel):
    name: str = Field(..., description="The name of the related item")
    weight: float = Field(..., description="The weight of the relation")

class VocabularyEntry(BaseModel):
    name: str = Field(..., description="The name of the term")
    r_isa: Optional[List[RelationItem]] = Field(default_factory=list, description="List of r_isa relations")
    r_raff_sem: Optional[List[RelationItem]] = Field(default_factory=list, description="List of r_raff_sem relations")
    r_pos: Optional[List[RelationItem]] = Field(default_factory=list, description="List of r_pos relations")

    @validator("r_isa", "r_raff_sem", pre=True)
    def clean_and_deduplicate(cls, value):
        if not value:
            return []

        cleaned = {}
        for v in value:
            name = v.get("name", "")
            weight = v.get("weight", 1.0)
        
            if name.startswith("en:"):
                continue
            if ">" in name:
                name = name.split(">")[0]
            if ":" in name:
                name = name.split(":")[0]
                
            name = name.lower()

            cleaned[name] = RelationItem(name=name, weight=weight)

        return list(cleaned.values())

    @validator("r_pos", pre=True)
    def clean_r_pos(cls, value):
        if not value:
            return []
        cleaned = []
        for v in value:
            name = v.get("name", "")
            weight = v.get("weight", 0.0)

            if name.startswith("Gender:"):
                continue

            if ":" in name:
                name = name.split(":")[0]

            name = name.lower().strip()
            if not name:
                continue

            cleaned.append(RelationItem(name=name, weight=weight))

        return cleaned


    def deduplicate_relations(self):
        def dedup(items: List[RelationItem]) -> List[RelationItem]:
            unique = {}
            for item in items:
                if item.name not in unique or item.weight > unique[item.name].weight:
                    unique[item.name] = item
            return list(unique.values())

        self.r_isa = dedup(self.r_isa)
        self.r_raff_sem = dedup(self.r_raff_sem)
        self.r_pos = dedup(self.r_pos)

class Vocabulary(BaseModel):
    data: List[VocabularyEntry] = Field(default_factory=list, description="List of vocabulary entries")

    @classmethod
    def from_json(cls, file_path: Path) -> 'Vocabulary':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data=[VocabularyEntry(**entry) for entry in data])

    def get_all_names_by_relation(self) -> Dict[str, List[str]]:
        relations_names = {
            'r_isa': set(),
            'r_raff_sem': set(),
            'r_pos': set()
        }

        for entry in self.data:
            for rel in entry.r_isa:
                relations_names['r_isa'].add(rel.name)
            for rel in entry.r_raff_sem:
                relations_names['r_raff_sem'].add(rel.name)
            for rel in entry.r_pos:
                relations_names['r_pos'].add(rel.name)
    
        print(len(relations_names['r_isa']), len(relations_names['r_raff_sem']), len(relations_names['r_pos']))
        return {k: list(v) for k, v in relations_names.items()}
    
    def save_names_by_relation(self, output_file: Path) -> None:
        names = self.get_all_names_by_relation()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(names, f, indent=2, ensure_ascii=False)
    
    
    def relation_stats(self, top: int = 15):
        counters = {
            'r_isa': Counter(),
            'r_raff_sem': Counter(),
            'r_pos': Counter()
        }

        for entry in self.data:
            for rel in entry.r_isa:
                counters['r_isa'][rel.name] += 1
            for rel in entry.r_raff_sem:
                counters['r_raff_sem'][rel.name] += 1
            for rel in entry.r_pos:
                counters['r_pos'][rel.name] += 1

        for rel_type, counter in counters.items():
            print(f"\n=== {rel_type.upper()} ===")
            print(f"Nb unique: {len(counter)} | Total occurences: {sum(counter.values())}")
            print(f"Top {top}:")
            for name, count in counter.most_common(top):
                print(f"  {name}: {count}")

        
            most_common = counter.most_common(top)

            if most_common:
                output_file = Path(f"data/vocabulary/rel_distrib/{top}/{rel_type}.png")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                labels, values = zip(*most_common)
                plt.figure(figsize=(10,5))
                plt.bar(labels, values)
                plt.xticks(rotation=45, ha="right")
                plt.title(f"Top {top} {rel_type}")
                plt.tight_layout()
                plt.savefig(output_file)



@app.command()
def words_space(vocab_path: Path, output_path: Path):
    vocabulary = Vocabulary.from_json(vocab_path)  

    vocabulary.save_names_by_relation(output_path)
    print(f"Names by relation saved to {output_path}")


@app.command()
def save_cleaned_vocabulary(vocab_path: Path, output_path: Path):
    vocabulary = Vocabulary.from_json(vocab_path)  

    for entry in vocabulary.data:
        entry.deduplicate_relations()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([entry.model_dump() for entry in vocabulary.data], f, ensure_ascii=False, indent=2)

    print(f"Cleaned vocabulary saved to {output_path}")



@app.command()
def relation_stats(vocab_path: Path, top: int):
    vocabulary = Vocabulary.from_json(vocab_path)
    vocabulary.relation_stats(top=top)


if __name__ == '__main__':
    app()