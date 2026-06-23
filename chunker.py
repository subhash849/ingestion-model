"""
chunker.py — Text preprocessing and paragraph-based chunking.

Design:
  - Paragraphs are the natural unit (double newline = boundary)
  - Chunks are assembled by accumulating paragraphs until CHUNK_MAX_TOKENS
  - Adjacent chunks share CHUNK_OVERLAP paragraphs so cross-boundary terms
    are never lost
  - Raw values (flight numbers, codes, dates) are stripped before chunking
    so the LLM focuses on concepts
"""

from __future__ import annotations
import re
import logging
from config import RAW_VALUE_PATTERNS, CHUNK_MAX_TOKENS, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

_RAW_VALUE_RE = re.compile("|".join(RAW_VALUE_PATTERNS))

# Rough token estimator: 1 token ≈ 4 characters (good enough for budgeting)
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    """
    Normalize whitespace and strip raw values.
    Returns cleaned text or empty string.
    """
    if not text or not text.strip():
        return ""
    text = re.sub(r"[ \t]+", " ", text)           # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse excess blank lines
    text = _RAW_VALUE_RE.sub(" ", text)            # remove raw values
    text = re.sub(r" {2,}", " ", text)             # clean up leftover spaces
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """
    Split cleaned text into overlapping paragraph-based chunks.

    Each chunk stays under CHUNK_MAX_TOKENS.
    Adjacent chunks share CHUNK_OVERLAP paragraphs for context continuity.

    Returns a list of chunk strings (never empty strings).
    """
    # Split on blank lines (paragraph boundaries)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        # Single paragraph already exceeds limit — split it by sentences
        if para_tokens > CHUNK_MAX_TOKENS:
            if current:
                chunks.append("\n\n".join(current))
                current = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
                current_tokens = sum(_estimate_tokens(p) for p in current)
            # Hard-split oversized paragraph by sentences
            for sentence_chunk in _split_paragraph(para):
                chunks.append(sentence_chunk)
            continue

        # Would exceed limit — flush current, start new with overlap
        if current_tokens + para_tokens > CHUNK_MAX_TOKENS and current:
            chunks.append("\n\n".join(current))
            # Keep last N paragraphs as overlap for the next chunk
            overlap = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
            current = overlap + [para]
            current_tokens = sum(_estimate_tokens(p) for p in current)
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    logger.debug("Chunked text into %d chunks", len(chunks))
    return chunks


def _split_paragraph(para: str) -> list[str]:
    """
    Last-resort splitter for oversized single paragraphs.
    Splits on sentence boundaries and re-assembles into token-safe pieces.
    """
    sentences = re.split(r"(?<=[.!?])\s+", para)
    result: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        t = _estimate_tokens(sent)
        if current_tokens + t > CHUNK_MAX_TOKENS and current:
            result.append(" ".join(current))
            current = [sent]
            current_tokens = t
        else:
            current.append(sent)
            current_tokens += t

    if current:
        result.append(" ".join(current))
    return result
