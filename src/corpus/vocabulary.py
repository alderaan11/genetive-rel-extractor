import json
from pathlib import Path
from typing import List, Dict, Any
import typer

app = typer.Typer()

def load_vocabulary(vocabulary_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load enriched vocabulary JSON into a dictionary keyed by term name."""
    with open(vocabulary_path, 'r', encoding='utf-8') as f:
        vocabulary_list = json.load(f)
    vocabulary_dict = {}
    for entry in vocabulary_list:
        if 'name' in entry:
            term_name = entry['name'].split('>')[0]
            vocabulary_dict[term_name] = entry
        else:
            print(f"Warning: Skipping vocabulary entry without 'name' field: {entry}")
    return vocabulary_dict


def transform_entry(entry: Dict[str, Any], vocabulary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Transform a single corpus entry, populating r_isa, r_raff, and _pos."""
    new_entry = {
        "prep": entry.get("prep", ""),
        "rel_type": entry.get("rel_type", ""),
        "det": entry.get("det", False)
    }
    
    for term_key in ["termA", "termB"]:
        term_data = entry.get(term_key, {})
        term_name = term_data.get("name", "").split('>')[0]  
        new_term_data = {"name": term_name}
        
        vocab_entry = vocabulary.get(term_name, {})
        
        
        new_term_data["r_isa"] = vocab_entry.get("r_isa", [])
        new_term_data["r_raff_sem"] = vocab_entry.get("r_raff_sem", [])  
        new_term_data["r_pos"] = vocab_entry.get("r_pos", [])
        
        new_entry[term_key] = new_term_data
        
        if not vocab_entry:
            print(f"Warning: Term '{term_name}' not found in vocabulary")
    
    return new_entry

@app.command()
def process_corpus_relations(
    corpus_dir: Path = typer.Argument(..., help="Directory containing rel_type*.json files"),
    vocabulary_path: Path = typer.Argument(..., help="Path to enriched vocabulary JSON"),
    output_dir: Path = typer.Argument(..., help="Directory to save processed JSON files")
):
    """
    Process all rel_type*.json files in corpus_dir, transforming hypernym to r_isa, trt to r_raff,
    and adding _pos using data from vocabulary_path.

    Args:
        corpus_dir: Directory containing rel_type*.json files
        vocabulary_path: Path to enriched vocabulary JSON
        output_dir: Directory to save processed JSON files
    """

    output_dir.mkdir(parents=True, exist_ok=True)


    vocabulary = load_vocabulary(vocabulary_path)
    print(f"Loaded vocabulary with {len(vocabulary)} terms")

    corpus_files = list(corpus_dir.glob("*.json"))


    total_files = len(corpus_files)
    print(f"Found {total_files} corpus files to process")

    for i, corpus_file in enumerate(corpus_files, 1):
        print(f"Processing {i}/{total_files}: {corpus_file}")

        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)    
        
        processed_data = [transform_entry(entry, vocabulary) for entry in corpus_data]

        
        output_file = output_dir / corpus_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed file saved to {output_file}")

    print(f"All {total_files} corpus files processed and saved to {output_dir}")

if __name__ == '__main__':
    app()