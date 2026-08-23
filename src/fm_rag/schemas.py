from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentPage:
    document_id: str
    file_name: str
    file_path: Path
    page_number: int
    page_count: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextChunk:
    document_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StructuredMeasure:
    document_id: str
    file_name: str
    page_number: int
    page_type: str

    observation_period: str
    comparison_period: str | None

    entity_type: str
    entity_name: str

    indicator: str
    value: int | float
    unit: str

    source_bbox: tuple[
        float,
        float,
        float,
        float,
    ]

    extraction_method: str = "pdf_native_geometry"
    confidence: float = 1.0
