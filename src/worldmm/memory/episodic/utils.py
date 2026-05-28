from dataclasses import dataclass
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from hashlib import md5

class NerRawOutput(BaseModel):
    named_entities: List[str]

class TripleRawOutput(BaseModel):
    triples: List[List[str]]

class CombinedOpenIERawOutput(BaseModel):
    named_entities: List[str]
    triples: List[List[str]]

@dataclass
class NerOutput:
    chunk_id: str
    unique_entities: List[str]
    metadata: Dict[str, Any]

@dataclass
class TripleOutput:
    chunk_id: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


def compute_mdhash_id(content: str, prefix: Optional[str] = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()

def filter_invalid_triples(triples: List[List[str]]) -> List[List[str]]:
    """
    Filters out invalid and duplicate triples from a list of triples.

    A valid triple meets the following criteria:
    1. It contains exactly three elements.
    2. It is unique within the list (no duplicates in the output).

    The function ensures:
    - Each valid triple is converted to a list of strings.
    - The order of unique, valid triples is preserved.
    - Do not apply any text preprocessing techniques or rules within this function.
    
    Args:
        triples (List[List[str]]): 
            A list of triples (each a list of strings or elements that can be converted to strings).

    Returns:
        List[List[str]]: 
            A list of unique, valid triples, each represented as a list of strings.
    """
    unique_triples = set()
    valid_triples = []

    for triple in triples:
        if len(triple) != 3: continue  # Skip triples that do not have exactly 3 elements

        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)

    return valid_triples