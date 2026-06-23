"""
pipeline.py — Orchestrates the full glossary extraction pipeline.

Flow per document:
  PDF/text → preprocess → chunk → LLM (per chunk) → aggregate → postprocess

Flow per dataset:
  N documents → per-file pipeline → dataset result

Design notes:
  - Domain is detected in the FIRST chunk of each file, then cached for
    subsequent chunks (avoids repeated detection calls)
  - If first-chunk confidence < threshold → fallback to "general" for ALL chunks
  - Dataset-level domain cache: if a dataset_id is re-processed, we skip
    re-detection and reuse previously resolved domain
  - Each document fails independently — one bad PDF never blocks the batch
  - Cost tracking: token usage logged per document
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    OLLAMA_HOST,
    DOMAIN_CONFIDENCE_THRESHOLD,
    FALLBACK_DOMAIN,
)
from pdf_reader import extract_text_from_pdf
from chunker import preprocess, chunk_text
from llm_client import call_llm_for_chunk
from postprocessor import postprocess
from models import DocumentResult, DatasetResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset-level domain cache
# { dataset_id: detected_domain }
# Persists across multiple calls within the same process session.
# ---------------------------------------------------------------------------
_dataset_domain_cache: dict[str, str] = {}


def _confidence_to_float(confidence: str) -> float:
    return {"high": 0.9, "medium": 0.65, "low": 0.3}.get(confidence, 0.0)


# ---------------------------------------------------------------------------
# Single document processor
# ---------------------------------------------------------------------------

def process_document(
    doc_id: str,
    text: str,
    client: Any,
    domain_hint: str | None = None,
) -> DocumentResult:
    """
    Run the full pipeline on pre-extracted text for one document.

    Args:
        doc_id      : unique identifier for this document
        text        : raw extracted text (from PDF or plain input)
        client      : Groq or Ollama client instance
        domain_hint : pre-resolved domain (from dataset cache), or None

    Returns:
        DocumentResult with glossary, domain, confidence, and flags.
    """
    logger.info("[%s] Starting processing", doc_id)

    # ── Preprocess ──────────────────────────────────────────────────────────
    clean = preprocess(text)
    if not clean:
        logger.warning("[%s] Empty after preprocessing", doc_id)
        return DocumentResult(
            doc_id=doc_id, detected_domain=FALLBACK_DOMAIN,
            confidence="low", glossary=[], flagged=True,
            error="Document empty after preprocessing",
        )

    # ── Chunk ────────────────────────────────────────────────────────────────
    chunks = chunk_text(clean)
    if not chunks:
        return DocumentResult(
            doc_id=doc_id, detected_domain=FALLBACK_DOMAIN,
            confidence="low", glossary=[], flagged=True,
            error="No chunks produced from document",
        )

    logger.info("[%s] %d chunks to process", doc_id, len(chunks))

    # ── LLM extraction (per chunk) ────────────────────────────────────────
    all_entries: list[dict] = []
    resolved_domain    = domain_hint or FALLBACK_DOMAIN
    resolved_confidence = "low"
    domain_locked      = domain_hint is not None  # already known from cache

    for i, chunk in enumerate(chunks):
        # Only detect domain from the first chunk (unless already provided)
        hint = resolved_domain if domain_locked else None
        response = call_llm_for_chunk(chunk, client, domain_hint=hint)

        if response is None:
            logger.warning("[%s] Chunk %d/%d: LLM call failed, skipping", doc_id, i+1, len(chunks))
            continue

        # Lock domain after first successful chunk response
        if not domain_locked:
            conf_float = _confidence_to_float(response.confidence)
            if conf_float >= DOMAIN_CONFIDENCE_THRESHOLD:
                resolved_domain     = response.domain
                resolved_confidence = response.confidence
                domain_locked       = True
                logger.info("[%s] Domain locked: %s (confidence=%s)", doc_id, resolved_domain, resolved_confidence)
            else:
                resolved_domain     = FALLBACK_DOMAIN
                resolved_confidence = "low"
                domain_locked       = True
                logger.warning(
                    "[%s] Low domain confidence (%s) → falling back to 'general'",
                    doc_id, response.confidence,
                )

        # Collect glossary entries as plain dicts for postprocessor
        for entry in response.glossary:
            if hasattr(entry, "model_dump"):
                all_entries.append(entry.model_dump())
            elif hasattr(entry, "dict"):
                all_entries.append(entry.dict())
            else:
                all_entries.append(vars(entry))

    logger.info("[%s] Collected %d raw entries across all chunks", doc_id, len(all_entries))

    # ── Post-process ─────────────────────────────────────────────────────────
    final_glossary_dicts = postprocess(all_entries, resolved_domain)

    # Convert back to GlossaryEntry objects for the result model
    from models import GlossaryEntry, PYDANTIC_AVAILABLE
    if PYDANTIC_AVAILABLE:
        glossary_objects = [GlossaryEntry(**e) for e in final_glossary_dicts]
    else:
        glossary_objects = final_glossary_dicts  # type: ignore[assignment]

    flagged = resolved_confidence == "low" or resolved_domain == FALLBACK_DOMAIN

    logger.info(
        "[%s] Done — %d terms | domain=%s | confidence=%s | flagged=%s",
        doc_id, len(glossary_objects), resolved_domain, resolved_confidence, flagged,
    )

    return DocumentResult(
        doc_id          = doc_id,
        detected_domain = resolved_domain,
        confidence      = resolved_confidence,
        glossary        = glossary_objects,
        flagged         = flagged,
    )


# ---------------------------------------------------------------------------
# Public API — batch entry point
# ---------------------------------------------------------------------------

def run_pipeline(payload: dict) -> dict:
    """
    Public entry point. Accepts PDF file paths OR pre-extracted text content.

    Input schema:
    {
      "dataset_id": "string",
      "documents": [
        {
          "doc_id":   "string",
          "pdf_path": "path/to/file.pdf",   ← use this OR content
          "content":  "raw text string"      ← use this OR pdf_path
        }
      ]
    }

    Output schema mirrors DatasetResult.
    """
    dataset_id = payload.get("dataset_id", "default")
    documents  = payload.get("documents", [])

    if not isinstance(documents, list) or not documents:
        raise ValueError("`documents` must be a non-empty list.")

    if LLM_PROVIDER == "ollama":
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
    else:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

    # Check dataset-level domain cache
    cached_domain = _dataset_domain_cache.get(dataset_id)
    if cached_domain:
        logger.info("Dataset '%s': using cached domain '%s'", dataset_id, cached_domain)

    results: list[DocumentResult] = []

    for doc in documents:
        doc_id   = doc.get("doc_id", "unknown")
        pdf_path = doc.get("pdf_path")
        content  = doc.get("content", "")

        # ── Extract text ────────────────────────────────────────────────────
        if pdf_path:
            text, err = extract_text_from_pdf(pdf_path)
            if err:
                logger.error("[%s] PDF extraction failed: %s", doc_id, err)
                results.append(DocumentResult(
                    doc_id=doc_id, detected_domain=FALLBACK_DOMAIN,
                    confidence="low", glossary=[], flagged=True, error=err,
                ))
                continue
        elif content:
            text = content
        else:
            results.append(DocumentResult(
                doc_id=doc_id, detected_domain=FALLBACK_DOMAIN,
                confidence="low", glossary=[], flagged=True,
                error="No pdf_path or content provided",
            ))
            continue

        # ── Process ─────────────────────────────────────────────────────────
        result = process_document(
            doc_id      = doc_id,
            text        = text,
            client      = client,
            domain_hint = cached_domain,
        )
        results.append(result)

        # Update dataset domain cache from first successful high-confidence doc
        if (
            dataset_id not in _dataset_domain_cache
            and not result.flagged
            and result.detected_domain != FALLBACK_DOMAIN
        ):
            _dataset_domain_cache[dataset_id] = result.detected_domain
            logger.info(
                "Dataset '%s': domain cached as '%s'",
                dataset_id, result.detected_domain,
            )

    # ── Build dataset result ─────────────────────────────────────────────────
    dataset_result = DatasetResult(
        dataset_id      = dataset_id,
        results         = results,
        total_docs      = len(results),
        successful_docs = sum(1 for r in results if not r.error),
        flagged_docs    = sum(1 for r in results if r.flagged),
    )

    # Serialize to plain dict for JSON compatibility
    if hasattr(dataset_result, "model_dump"):
        return dataset_result.model_dump()
    elif hasattr(dataset_result, "dict"):
        return dataset_result.dict()
    else:
        return vars(dataset_result)
