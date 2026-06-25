"""
postprocessor.py — Merge, validate, normalize, and deduplicate glossary entries
collected across all chunks of a single document.

Deduplication strategy:
  - Exact match after snake_case normalization (fast, first-wins)
  - Near-duplicate detection via token overlap (catches blood_pressure vs
    arterial_blood_pressure style overlaps) — logged but NOT auto-merged,
    because merging requires semantic judgment better left to a human flag.

Nothing in this module makes LLM calls.
"""

from __future__ import annotations
import re
import logging
from collections import Counter
from config import RAW_VALUE_PATTERNS, DOMAIN_CATEGORIES, FALLBACK_DOMAIN

logger = logging.getLogger(__name__)

_RAW_VALUE_RE = re.compile("|".join(RAW_VALUE_PATTERNS))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_entry(entry: dict) -> bool:
    """
    Structural + content guard. Rejects:
      - Missing term or definition
      - Single-character terms
      - Terms that are raw values (slipped past preprocessing)
      - Terms that are purely numeric
    """
    term = entry.get("term", "").strip()
    defn = entry.get("definition", "").strip()

    if not term or not defn:
        return False
    if len(term) <= 1:
        return False
    if term.isdigit():
        return False
    if _RAW_VALUE_RE.fullmatch(term.upper()):
        return False
    return True


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _best_category_match(raw: str, valid_categories: list[str]) -> str:
    """
    Find the best matching valid category for a raw model-returned string.

    Strategy (in order):
      1. Exact match after normalization
      2. Substring match  (e.g. "flight_operations" contains "flight_ops" tokens)
      3. Token overlap    (Jaccard on underscore-split tokens)
      4. Fallback → "general"
    """
    raw_norm = re.sub(r"\s+", "_", raw.strip().lower())

    # 1. Exact
    if raw_norm in valid_categories:
        return raw_norm

    # 2. Substring: valid category appears inside the model's string or vice-versa
    for cat in valid_categories:
        if cat in raw_norm or raw_norm in cat:
            logger.debug("Category fuzzy match (substring): '%s' → '%s'", raw_norm, cat)
            return cat

    # 3. Token overlap (Jaccard)
    raw_tokens = set(raw_norm.split("_"))
    best_cat, best_score = "general", 0.0
    for cat in valid_categories:
        cat_tokens = set(cat.split("_"))
        score = len(raw_tokens & cat_tokens) / len(raw_tokens | cat_tokens) if (raw_tokens | cat_tokens) else 0.0
        if score > best_score:
            best_score = score
            best_cat   = cat

    if best_score >= 0.4:
        logger.debug("Category fuzzy match (Jaccard=%.2f): '%s' → '%s'", best_score, raw_norm, best_cat)
        return best_cat

    logger.warning("No category match for '%s' in %s → 'general'", raw_norm, valid_categories)
    return "general"


def normalize_entry(entry: dict, valid_categories: list[str]) -> dict:
    """
    Normalize a single glossary entry:
      - term       → snake_case, lowercase
      - definition → stripped
      - category   → fuzzy-matched against valid_categories; 'general' only as last resort
    """
    term     = re.sub(r"\s+", "_", entry["term"].strip().lower())
    defn     = entry["definition"].strip()
    raw_cat  = entry.get("category", "").strip().lower()
    category = _best_category_match(raw_cat, valid_categories)
    page_number = entry.get("page_number")
    chunk_number = entry.get("chunk_number")

    return {"term": term, "definition": defn, "category": category, "page_number": page_number, "chunk_number": chunk_number}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard similarity on word tokens. Used for near-duplicate detection."""
    ta = set(a.split("_"))
    tb = set(b.split("_"))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def deduplicate(entries: list[dict]) -> list[dict]:
    """
    Exact deduplication (first-wins) + near-duplicate logging.

    Near-duplicates (Jaccard ≥ 0.7) are logged as warnings so a human
    can review them — they are NOT auto-merged because the definitions
    may be meaningfully different across chunks.
    """
    seen_exact: dict[str, dict] = {}   # term → entry
    unique: list[dict] = []

    for entry in entries:
        key = entry["term"]
        if key in seen_exact:
            continue   # exact duplicate → drop

        # Near-duplicate check
        for existing_key in seen_exact:
            ratio = _token_overlap_ratio(key, existing_key)
            if ratio >= 0.7:
                logger.warning(
                    "Near-duplicate terms detected: '%s' ↔ '%s' (overlap=%.2f) — review recommended",
                    key, existing_key, ratio,
                )
                break

        seen_exact[key] = entry
        unique.append(entry)

    logger.debug("Deduplication: %d → %d entries", len(entries), len(unique))
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess(
    all_entries: list[dict],
    domain: str,
) -> list[dict]:
    """
    Full post-processing pipeline for all entries collected from a document's chunks.

    Steps: validate → normalize (domain-aware categories) → deduplicate

    Args:
        all_entries : flat list of raw dicts from all chunk LLM responses
        domain      : detected domain (used to look up valid categories)

    Returns:
        Clean, deduplicated, normalized list of glossary entry dicts.
    """
    valid_categories = DOMAIN_CATEGORIES.get(domain, DOMAIN_CATEGORIES[FALLBACK_DOMAIN])

    validated  = [e for e in all_entries if validate_entry(e)]
    normalized = [normalize_entry(e, valid_categories) for e in validated]
    final      = deduplicate(normalized)

    logger.info(
        "Postprocess: %d raw → %d valid → %d after dedup (domain=%s)",
        len(all_entries), len(validated), len(final), domain,
    )
    return final