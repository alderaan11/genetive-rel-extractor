from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

class Prep(str, Enum):
    DE = "DE"
    DES = "DES"
    D = "D'"
    DU = "DU"

class Article(str, Enum):
    LA = "LA"
    LE = "LE"
    LES = "LES" 
    L = "L'"
    UN = "UN"
    UNE = "UNE"

class TermInfo(BaseModel):
    name: str
    r_isa: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    r_raff_sem: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    r_pos: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class RelationInstance(BaseModel):
    termA: TermInfo
    termB: TermInfo
    prep: Prep 
    relation_type: str 
    is_det: bool
    determinant: Optional[Article] = None  


class Corpus(BaseModel):
    original_file: Optional[Path] = None
    data: Dict[str, RelationInstance] = Field(default_factory=dict)

