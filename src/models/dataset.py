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
    hypernym: Optional[Dict[str, float]] = Field(default_factory=dict) #r_isa n° 6
    trt: Optional[List[str]] = Field(default_factory=list)


class Relation(BaseModel):
    termA: Term
    termB: Term
    prep: Prep 
    rel_type: str 
   


class Corpus(BaseModel):
    data: List[Relation] = Field(default_factory=list)