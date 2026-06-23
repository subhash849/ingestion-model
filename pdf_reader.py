"""
pdf_reader.py — PDF → clean text extraction with quality checks.

Handles:
  - Normal text-based PDFs       (pdfplumber)
  - Scanned / image-only PDFs    (detected, flagged — OCR deferred)
  - Encrypted PDFs               (detected, skipped gracefully)
  - Corrupted / unreadable files (caught, error returned)

Returns a plain string. Callers never see PDF internals.
"""

from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum characters to consider a page "text-bearing"
_MIN_CHARS_PER_PAGE = 30


def extract_text_from_pdf(path: str | Path) -> tuple[str, str | None]:
    """
    Extract text from a PDF file.

    Returns:
        (text, error)
        - text  : extracted string (may be empty on failure)
        - error : None on success, human-readable message on failure
    """
    try:
        import pdfplumber
    except ImportError:
        return "", "pdfplumber not installed — run: pip install pdfplumber"

    path = Path(path)
    if not path.exists():
        return "", f"File not found: {path}"
    if path.suffix.lower() != ".pdf":
        return "", f"Not a PDF file: {path}"

    try:
        with pdfplumber.open(path) as pdf:

            # Encrypted PDF guard
            if pdf.doc.is_encrypted:            # type: ignore[attr-defined]
                return "", f"PDF is encrypted and cannot be read: {path.name}"

            pages_text: list[str] = []
            scanned_pages = 0

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < _MIN_CHARS_PER_PAGE:
                    scanned_pages += 1
                    logger.debug("Page %d appears scanned or image-only", i + 1)
                else:
                    pages_text.append(text)

            total_pages = len(pdf.pages)

            # If more than 80% of pages are image-only → flag as scanned
            if total_pages > 0 and (scanned_pages / total_pages) > 0.8:
                return "", (
                    f"PDF appears to be scanned ({scanned_pages}/{total_pages} pages "
                    f"have no extractable text). OCR required: {path.name}"
                )

            full_text = "\n\n".join(pages_text).strip()
            if not full_text:
                return "", f"No text could be extracted from: {path.name}"

            logger.info(
                "Extracted %d chars from %d/%d pages of '%s'",
                len(full_text), len(pages_text), total_pages, path.name,
            )
            return full_text, None

    except Exception as exc:      # pdfplumber raises various internal errors
        return "", f"Failed to read PDF '{path.name}': {exc}"
