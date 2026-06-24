"""
pipeline.py — Orchestrates the full glossary extraction pipeline.
"""

from __future__ import annotations
import logging
from typing import Optional
from pathlib import Path
from groq import Groq

from config import (
    GROQ_API_KEY,
    DOMAIN_CONFIDENCE_THRESHOLD,
    DOMAIN_DETECTION_CHUNKS,
    FALLBACK_DOMAIN,
)
from pdf_reader import extract_text_from_pdf
from chunker import preprocess, chunk_text
from llm_client import call_llm_for_chunk
from postprocessor import postprocess
from models import DocumentResult, DatasetResult

logger = logging.getLogger(__name__)

_dataset_domain_cache: dict[str, str] = {}


def _confidence_to_float(confidence: str) -> float:
    return {"high": 0.9, "medium": 0.65, "low": 0.3}.get(confidence, 0.0)


def _majority_domain(votes: list[tuple[str, str]]) -> tuple[str, str]:
    """
    Pick the domain that appears most frequently across votes.
    Weighted by confidence: high=3, medium=2, low=1.
    Falls back to FALLBACK_DOMAIN if no vote clears DOMAIN_CONFIDENCE_THRESHOLD.
    """
    weights = {"high": 3, "medium": 2, "low": 1}
    scores: dict[str, float] = {}
    for domain, confidence in votes:
        scores[domain] = scores.get(domain, 0) + weights.get(confidence, 1)

    if not scores:
        return FALLBACK_DOMAIN, "low"

    best_domain = max(scores, key=lambda d: scores[d])

    # Find the highest confidence vote for the winning domain
    best_conf = "low"
    for domain, confidence in votes:
        if domain == best_domain:
            if _confidence_to_float(confidence) > _confidence_to_float(best_conf):
                best_conf = confidence

    # Reject if best domain never got a confident vote
    if _confidence_to_float(best_conf) < DOMAIN_CONFIDENCE_THRESHOLD:
        return FALLBACK_DOMAIN, "low"

    return best_domain, best_conf


def process_document(
    doc_id: str,
    text: str,
    client: Groq,
    domain_hint: Optional[str] = None,
) -> DocumentResult:

    logger.info("[%s] Starting processing", doc_id)

    clean = preprocess(text)
    if not clean:
        logger.warning("[%s] Empty after preprocessing", doc_id)
        return DocumentResult(
            doc_id=doc_id, detected_domain=FALLBACK_DOMAIN,
            confidence="low", glossary=[], flagged=True,
            error="Document empty after preprocessing",
        )

    chunks = chunk_text(clean)
    if not chunks:
        return DocumentResult(
            doc_id=doc_id, detected_domain=FALLBACK_DOMAIN,
            confidence="low", glossary=[], flagged=True,
            error="No chunks produced from document",
        )

    logger.info("[%s] %d chunks to process", doc_id, len(chunks))

    all_entries: list[dict] = []
    resolved_domain     = domain_hint or FALLBACK_DOMAIN
    resolved_confidence = "low"
    domain_locked       = domain_hint is not None
    domain_votes: list[tuple[str, str]] = []

    for i, chunk in enumerate(chunks):
        hint = resolved_domain if domain_locked else None
        response = call_llm_for_chunk(chunk, client, domain_hint=hint)

        if response is None:
            logger.warning("[%s] Chunk %d/%d: LLM call failed, skipping", doc_id, i+1, len(chunks))
            continue

        # ── Domain voting: accumulate across first N chunks ─────────────────
        if not domain_locked:
            domain_votes.append((response.domain, response.confidence))
            logger.debug(
                "[%s] Domain vote %d: %s (confidence=%s)",
                doc_id, len(domain_votes), response.domain, response.confidence,
            )

            # Lock when enough votes collected OR last chunk reached
            if len(domain_votes) >= DOMAIN_DETECTION_CHUNKS or i == len(chunks) - 1:
                resolved_domain, resolved_confidence = _majority_domain(domain_votes)
                domain_locked = True
                if resolved_domain == FALLBACK_DOMAIN:
                    logger.warning(
                        "[%s] No confident domain from %d votes → 'general'",
                        doc_id, len(domain_votes),
                    )
                else:
                    logger.info(
                        "[%s] Domain locked after %d votes: %s (confidence=%s)",
                        doc_id, len(domain_votes), resolved_domain, resolved_confidence,
                    )

        # ── Collect entries ──────────────────────────────────────────────────
        for entry in response.glossary:
            if hasattr(entry, "model_dump"):
                all_entries.append(entry.model_dump())
            elif hasattr(entry, "dict"):
                all_entries.append(entry.dict())
            else:
                all_entries.append(vars(entry))

    logger.info("[%s] Collected %d raw entries across all chunks", doc_id, len(all_entries))

    final_glossary_dicts = postprocess(all_entries, resolved_domain)

    from models import GlossaryEntry, PYDANTIC_AVAILABLE
    if PYDANTIC_AVAILABLE:
        glossary_objects = [GlossaryEntry(**e) for e in final_glossary_dicts]
    else:
        glossary_objects = final_glossary_dicts

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


def run_pipeline(payload: dict) -> dict:
    dataset_id = payload.get("dataset_id", "default")
    documents  = payload.get("documents", [])

    if not isinstance(documents, list) or not documents:
        raise ValueError("`documents` must be a non-empty list.")

    client = Groq(api_key=GROQ_API_KEY)

    cached_domain = _dataset_domain_cache.get(dataset_id)
    if cached_domain:
        logger.info("Dataset '%s': using cached domain '%s'", dataset_id, cached_domain)

    results: list[DocumentResult] = []

    for doc in documents:
        doc_id   = doc.get("doc_id", "unknown")
        pdf_path = doc.get("pdf_path")
        content  = doc.get("content", "")

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

        result = process_document(
            doc_id      = doc_id,
            text        = text,
            client      = client,
            domain_hint = cached_domain,
        )
        results.append(result)

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

    dataset_result = DatasetResult(
        dataset_id      = dataset_id,
        results         = results,
        total_docs      = len(results),
        successful_docs = sum(1 for r in results if not r.error),
        flagged_docs    = sum(1 for r in results if r.flagged),
    )

    if hasattr(dataset_result, "model_dump"):
        return dataset_result.model_dump()
    elif hasattr(dataset_result, "dict"):
        return dataset_result.dict()
    else:
        return vars(dataset_result)