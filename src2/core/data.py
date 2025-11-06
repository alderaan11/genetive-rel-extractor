from pathlib import Path
import json
from typing import List
from src2.models import RelationInstance


def load_json_corpus(file_path: Path) -> List[RelationInstance]:
    """Charge un fichier JSON et retourne une liste de RelationInstance."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [RelationInstance(**v) for v in data.get("data", {}).values()]