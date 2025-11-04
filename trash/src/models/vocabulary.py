from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics
import typer
from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, iqr  # Pour skewness et IQR


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
        counters = self._get_relation_counters()
        print("Relation statistics:")

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

    def log_transform_weights(self):
        all_weights = self._collect_all_weights()
        
        if not all_weights:
            print("Aucun poids à transformer.")
            return
        
        weights_array = np.array(all_weights)
        transformed_weights = np.log1p(weights_array)  # log(1 + x)
        
        mean = np.mean(transformed_weights)
        st_dev = np.std(transformed_weights) if len(transformed_weights) > 1 else 1.0
        
        def norm_func(w):
            transformed_w = np.log1p(w)  
            return (transformed_w - mean) / st_dev if st_dev > 0 else 0.0
        
        for entry in self.data:
            for rel in entry.r_isa + entry.r_raff_sem + entry.r_pos:
                rel.weight = norm_func(rel.weight)

    def normalize_weights(self):
        for entry in self.data:
            all_weights = [
                rel.weight for rel in entry.r_isa
            ] + [rel.weight for rel in entry.r_raff_sem
            ] + [
                rel.weight for rel in entry.r_pos
            ]
            
        mean = statistics.mean(all_weights) if all_weights else 1.0
        stdev = statistics.stdev(all_weights) if len(all_weights) > 1 else 0.0 

        def normalize_weights(weight: float) -> float:
            if stdev == 0:
                return 0.0
            return (weight - mean) / stdev
        
        for entry in self.data:
                for rel in entry.r_isa:
                    rel.weight = normalize_weights(rel.weight)
                for rel in entry.r_raff_sem:
                    rel.weight = normalize_weights(rel.weight)
                for rel in entry.r_pos:
                    rel.weight = normalize_weights(rel.weight)
       
    def _collect_all_weights(self) -> List[float]:
        weights = []
        for entry in self.data:
            weights.extend([rel.weight for rel in entry.r_isa])
            weights.extend([rel.weight for rel in entry.r_raff_sem])
            weights.extend([rel.weight for rel in entry.r_pos])
        return weights    


    
    def _get_relation_counters(self) -> Dict[str, Counter]:
        counters = {'r_isa': Counter(), 'r_raff_sem': Counter(), 'r_pos': Counter()}
        for entry in self.data:
            for rel in entry.r_isa: counters['r_isa'][rel.name] += 1
            for rel in entry.r_raff_sem: counters['r_raff_sem'][rel.name] += 1
            for rel in entry.r_pos: counters['r_pos'][rel.name] += 1
        return counters

    def analyze_weights(self, output_path: Path):
        all_weights = np.array(self._collect_all_weights())

        if all_weights.size == 0:
            print("No weights to analyze.")
            return

        mean = np.mean(all_weights)
        median = np.median(all_weights)
        std_dev = np.std(all_weights)
        min_w = np.min(all_weights)
        max_w = np.max(all_weights)
        skewness = skew(all_weights)
        iqr_value = iqr(all_weights)

        print("Analyse des poids :")
        print(f"- Nombre : {len(all_weights)}")
        print(f"- Moyenne : {mean:.2f}")
        print(f"- Médiane : {median:.2f}")
        print(f"- Écart-type : {std_dev:.2f}")
        print(f"- Asymétrie (skewness) : {skewness:.2f}")
        print(f"- Min/Max : {min_w:.2f} / {max_w:.2f}")
        print(f"- IQR : {iqr_value:.2f}") #IQR is the difference between the 75th and 25th percentiles
        
        if skewness > 1:
            print("-> Hautement asymétrique à droite (right-skewed). Considérez log_transform.")
        elif skewness > 0.5:
            print("-> Modérément asymétrique à droite.")
        elif skewness < -1:
            print("-> Hautement asymétrique à gauche (left-skewed).")
        elif skewness < -0.5:
            print("-> Modérément asymétrique à gauche.")
        else:
            print("-> Approximativement symétrique.")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.hist(all_weights, bins=20)
        plt.title("Distribution des poids")
        plt.xlabel("Poids")
        plt.ylabel("Fréquence")
        plt.savefig(output_path)
        print(f"Histogramme sauvegardé : {output_path}")



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
def analyze_weights(vocab_path: Path, output_path: Path):
    vocabulary = Vocabulary.from_json(vocab_path)
    vocabulary.analyze_weights(output_path)

@app.command()
def relation_stats(vocab_path: Path, top: int):
    vocabulary = Vocabulary.from_json(vocab_path)
    vocabulary.relation_stats(top=top)

@app.command()
def normalize_weights(vocab_path: Path, output_path: Path):
    vocabulary = Vocabulary.from_json(vocab_path)
    vocabulary.log_transform_weights()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([entry.model_dump() for entry in vocabulary.data], f, ensure_ascii=False, indent=2)

    print(f"Vocabulary with normalized weights saved to {output_path}")

if __name__ == '__main__':
    app()