from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class Prep(str, Enum):
    DE = "DE"
    DES = "DES"
    D = "D'"
    DU = "DU"

class Term(BaseModel):
    name: str
    r_isa: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    r_raff_sem: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    r_pos: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class Relation(BaseModel):
    termA: Term
    termB: Term
    prep: Prep 
    rel_type: str 
    det: bool
   


class Corpus(BaseModel):
    data: List[Relation] = Field(default_factory=list)