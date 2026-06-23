"""
models.py — Pydantic schemas for every data structure in the pipeline.
All LLM responses are validated here before entering post-processing.
If Pydantic is not installed the pipeline falls back to plain dicts (see note).
"""

from __future__ import annotations

try:
    from pydantic import BaseModel, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object                          # type: ignore[assignment,misc]
    def field_validator(*args, **kwargs):       # type: ignore[misc]
        def _noop(fn): return fn
        return _noop


# ---------------------------------------------------------------------------
# LLM response schemas  (validated immediately after JSON parse)
# ---------------------------------------------------------------------------

class GlossaryEntry(BaseModel):
    term:       str
    definition: str
    category:   str

    @field_validator("term")
    @classmethod
    def term_must_be_snake(cls, v: str) -> str:
        import re
        return re.sub(r"\s+", "_", v.strip().lower())

    @field_validator("category")
    @classmethod
    def category_lowercase(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("definition")
    @classmethod
    def definition_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("definition must not be empty")
        return v


class ChunkLLMResponse(BaseModel):
    """Single-call response: domain + glossary in one shot."""
    domain:     str
    confidence: str          # "high" | "medium" | "low"
    glossary:   list[GlossaryEntry]

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("high", "medium", "low"):
            return "low"
        return v

    @field_validator("domain")
    @classmethod
    def domain_lowercase(cls, v: str) -> str:
        return v.strip().lower()


# ---------------------------------------------------------------------------
# Pipeline output schemas
# ---------------------------------------------------------------------------

class DocumentResult(BaseModel):
    doc_id:          str
    detected_domain: str
    confidence:      str
    glossary:        list[GlossaryEntry]
    flagged:         bool = False       # True when confidence < threshold
    error:           str | None = None  # populated on processing failure


class DatasetResult(BaseModel):
    dataset_id: str
    results:    list[DocumentResult]
    total_docs:         int
    successful_docs:    int
    flagged_docs:       int
