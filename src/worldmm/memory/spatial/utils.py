from dataclasses import dataclass
from pydantic import BaseModel, field_validator
from typing import List, Optional


SPATIAL_PREDICATE_VOCAB = frozenset({
    "in",
    "on",
    "under",
    "next_to",
    "near",
    "left_of",
    "right_of",
    "behind",
    "in_front_of",
    "contains",
    "located_in",
})


class SpatialRawOutput(BaseModel):
    spatial_triples: List[List[str]]
    episodic_evidence: List[List[int]]

    @field_validator("spatial_triples")
    def _validate_triples(cls, v):
        clean: List[List[str]] = []
        for t in v:
            if len(t) != 3:
                continue
            subj, pred, obj = t
            if not isinstance(pred, str):
                continue
            if pred not in SPATIAL_PREDICATE_VOCAB:
                continue
            if not subj or not obj:
                continue
            clean.append([subj.strip(), pred.strip(), obj.strip()])
        return clean


class SpatialConsolidationRawOutput(BaseModel):
    updated_triple: List[str]
    triples_to_remove: List[int]

    @field_validator("updated_triple")
    def _validate_updated_triple(cls, v):
        if len(v) != 3:
            raise ValueError("Updated triple must contain exactly 3 elements.")
        if v[1] not in SPATIAL_PREDICATE_VOCAB:
            raise ValueError(f"Predicate {v[1]!r} not in spatial vocab.")
        return v

    @field_validator("triples_to_remove")
    def _validate_triples_to_remove(cls, v):
        if not all(isinstance(i, int) for i in v):
            raise ValueError("All indices in triples_to_remove must be integers.")
        return v


@dataclass
class SpatialOutput:
    chunk_id: str
    spatial_triples: List[List[str]]
    episodic_evidence: List[List[int]]
