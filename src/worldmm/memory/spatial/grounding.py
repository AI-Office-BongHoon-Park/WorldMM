"""Geometric grounding schema for spatial triples."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GeometricGrounding(BaseModel):
    """Relative or metric 3D grounding for one side of a spatial triple."""

    model_config = ConfigDict(extra="forbid")

    bbox_center: list[float] = Field(..., min_length=3, max_length=3)
    bbox_extent: list[float] = Field(..., min_length=3, max_length=3)
    units: Literal["meters", "relative"]
    source: str
    confidence: float = Field(..., ge=-1.0, le=1.0)
    keyframe_ts: int
    instance_disambiguation: Literal[
        "single_match",
        "closest_in_time",
        "max_confidence",
        "tied_dropped",
    ]

    @field_validator("bbox_center", "bbox_extent")
    @classmethod
    def _validate_vec3(cls, value: list[float]) -> list[float]:
        return [float(item) for item in value]

    @model_validator(mode="after")
    def _validate_units_source(self) -> "GeometricGrounding":
        metric_source = "vggt_intrinsics" in self.source or "colmap_intrinsics" in self.source
        metric_depth = "metric" in self.source.lower()
        if self.units == "meters" and not (metric_source and metric_depth):
            raise ValueError("units='meters' requires metric depth and VGGT/COLMAP intrinsics in source")
        return self
